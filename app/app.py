import os
import json
import time
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, send_file, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, DecimalField
from wtforms.validators import DataRequired, NumberRange
from werkzeug.utils import secure_filename
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import plotly.graph_objects as go


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_CSV = os.path.join(BASE_DIR, 'static', 'results.csv')
DEMO_CSV = os.path.join(UPLOAD_FOLDER, 'Market_Basket_Optimisation.csv')
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class UploadForm(FlaskForm):
    file = FileField('CSV File', validators=[DataRequired()])
    min_support = DecimalField('Minimum Support (%)', default=2.0,
                               validators=[DataRequired(), NumberRange(min=0.01, max=100.0)])
    min_confidence = DecimalField('Minimum Confidence (%)', default=20.0,
                                  validators=[DataRequired(), NumberRange(min=0.0, max=100.0)])
    submit = SubmitField('Analyze')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_old_uploads(max_age_seconds=86400):
    """Delete uploaded files older than max_age_seconds (default 24h)."""
    now = time.time()
    for fname in os.listdir(UPLOAD_FOLDER):
        fpath = os.path.join(UPLOAD_FOLDER, fname)
        if os.path.isfile(fpath) and fpath != DEMO_CSV:
            if now - os.path.getmtime(fpath) > max_age_seconds:
                try:
                    os.remove(fpath)
                except OSError:
                    pass


def read_and_analyze(file_path, min_support, min_confidence):
    try:
        df = pd.read_csv(file_path, header=None, encoding='utf-8', on_bad_lines='skip')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, header=None, encoding='latin-1', on_bad_lines='skip')

    transactions = df.fillna('').astype(str).values.tolist()
    transactions = [[item.strip() for item in row if item.strip()] for row in transactions]
    transactions = [t for t in transactions if t]

    # Dataset statistics
    num_transactions = len(transactions)
    all_items = [item for t in transactions for item in t]
    unique_items = len(set(all_items))
    avg_basket = round(len(all_items) / num_transactions, 2) if num_transactions else 0
    from collections import Counter
    top_items = [item for item, _ in Counter(all_items).most_common(5)]

    dataset_stats = {
        'num_transactions': num_transactions,
        'unique_items': unique_items,
        'avg_basket': avg_basket,
        'top_items': top_items,
    }

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_).astype(bool)

    frequent_itemsets = apriori(df_encoded, min_support=min_support / 100, use_colnames=True)

    if frequent_itemsets.empty:
        return None, None, "No frequent itemsets found. Try lowering minimum support."

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence / 100)

    if rules.empty:
        return None, None, "No rules found. Try lowering minimum support (e.g. 1–2%) and/or minimum confidence."

    denom = 1 - rules['consequent support']
    rules['zhangs_metric'] = np.where(denom == 0, np.nan, rules['lift'] / denom - 1)

    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))

    for col in ['support', 'confidence', 'lift', 'leverage', 'conviction', 'zhangs_metric']:
        rules[col] = rules[col].round(4)

    return rules, dataset_stats, None


def create_charts(rules):
    bar = px.bar(
        rules.head(20), x='antecedents', y='support',
        title='Top 20 Rules — Antecedent Support',
        labels={'antecedents': 'Antecedent', 'support': 'Support'},
        color='confidence', color_continuous_scale='Blues'
    )
    bar.update_layout(xaxis_tickangle=-40, margin=dict(b=120))

    scatter = px.scatter(
        rules, x='support', y='confidence', color='lift',
        hover_data=['antecedents', 'consequents'],
        title='Support vs Confidence (colored by Lift)',
        labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'},
        color_continuous_scale='Viridis'
    )

    return bar.to_html(full_html=False), scatter.to_html(full_html=False)


def create_network_chart(rules, top_n=30):
    """Build a node-link network graph of the top N rules by lift."""
    top = rules.nlargest(top_n, 'lift')

    # Collect unique nodes
    nodes = list(set(top['antecedents'].tolist() + top['consequents'].tolist()))
    node_idx = {n: i for i, n in enumerate(nodes)}

    edge_x, edge_y = [], []
    node_x = [None] * len(nodes)
    node_y = [None] * len(nodes)

    # Position nodes in a circle
    import math
    n = len(nodes)
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        node_x[i] = math.cos(angle)
        node_y[i] = math.sin(angle)

    # Build edges
    for _, row in top.iterrows():
        xi = node_x[node_idx[row['antecedents']]]
        yi = node_y[node_idx[row['antecedents']]]
        xj = node_x[node_idx[row['consequents']]]
        yj = node_y[node_idx[row['consequents']]]
        edge_x += [xi, xj, None]
        edge_y += [yi, yj, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1, color='#cbd5e1'),
        hoverinfo='none'
    )

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=nodes, textposition='top center',
        marker=dict(size=14, color='#3b82f6', line=dict(width=2, color='white')),
        hoverinfo='text'
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f'Item Association Network (Top {top_n} rules by Lift)',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
        )
    )
    return fig.to_html(full_html=False)


def save_history_entry(filename, min_support, min_confidence, num_rules):
    """Append a compact entry to session analysis history (max 5)."""
    history = session.get('history', [])
    entry = {
        'filename': filename,
        'min_support': min_support,
        'min_confidence': min_confidence,
        'num_rules': num_rules,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
    history.insert(0, entry)
    session['history'] = history[:5]


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    error = None
    if form.validate_on_submit():
        file = form.file.data
        if not allowed_file(file.filename):
            error = 'Only CSV files are accepted.'
        else:
            cleanup_old_uploads()
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            min_support = float(form.min_support.data)
            min_confidence = float(form.min_confidence.data)
            rules, dataset_stats, err = read_and_analyze(file_path, min_support, min_confidence)
            if err:
                error = err
            else:
                rules.to_csv(RESULTS_CSV, index=False)
                session['last_file'] = file_path
                session['last_support'] = min_support
                session['last_confidence'] = min_confidence
                session['dataset_stats'] = dataset_stats
                save_history_entry(filename, min_support, min_confidence, len(rules))
                return redirect(url_for('results'))
    return render_template('index.html', form=form, error=error, history=session.get('history', []))


@app.route('/demo')
def demo():
    """Run analysis on the built-in demo dataset."""
    if not os.path.exists(DEMO_CSV):
        return redirect(url_for('index'))
    min_support, min_confidence = 1.0, 20.0
    rules, dataset_stats, err = read_and_analyze(DEMO_CSV, min_support, min_confidence)
    if err:
        return redirect(url_for('index'))
    rules.to_csv(RESULTS_CSV, index=False)
    session['last_file'] = DEMO_CSV
    session['last_support'] = min_support
    session['last_confidence'] = min_confidence
    session['dataset_stats'] = dataset_stats
    save_history_entry('demo_dataset.csv', min_support, min_confidence, len(rules))
    return redirect(url_for('results'))


@app.route('/rerun', methods=['POST'])
def rerun():
    """Re-run analysis on the last uploaded file with new thresholds."""
    file_path = session.get('last_file')
    if not file_path or not os.path.exists(file_path):
        return redirect(url_for('index'))
    try:
        min_support = float(request.form.get('min_support', 5))
        min_confidence = float(request.form.get('min_confidence', 50))
        min_support = max(0.01, min(100.0, min_support))
        min_confidence = max(0.0, min(100.0, min_confidence))
    except ValueError:
        return redirect(url_for('results'))
    rules, dataset_stats, err = read_and_analyze(file_path, min_support, min_confidence)
    if err:
        return redirect(url_for('results') + f'?error={err}')
    rules.to_csv(RESULTS_CSV, index=False)
    session['last_support'] = min_support
    session['last_confidence'] = min_confidence
    session['dataset_stats'] = dataset_stats
    save_history_entry(os.path.basename(file_path), min_support, min_confidence, len(rules))
    return redirect(url_for('results'))


@app.route('/results')
def results():
    try:
        df = pd.read_csv(RESULTS_CSV)

        # Metric filters
        try:
            min_lift = float(request.args.get('min_lift', 0))
            max_lift = float(request.args.get('max_lift', 9999))
            min_conf = float(request.args.get('min_conf', 0))
            max_conf = float(request.args.get('max_conf', 1))
            min_sup  = float(request.args.get('min_sup', 0))
        except ValueError:
            min_lift, max_lift, min_conf, max_conf, min_sup = 0, 9999, 0, 1, 0

        # Apply filters before search/pagination counts
        filtered = df[
            (df['lift'] >= min_lift) &
            (df['lift'] <= max_lift) &
            (df['confidence'] >= min_conf) &
            (df['confidence'] <= max_conf) &
            (df['support'] >= min_sup)
        ]
        total = len(filtered)

        query = request.args.get('search', '')
        if query:
            filtered = filtered[filtered.apply(
                lambda row: row.astype(str).str.contains(query, case=False, na=False).any(), axis=1
            )]

        page = int(request.args.get('page', 1))
        per_page = 50
        offset = (page - 1) * per_page
        shown_df = filtered.iloc[offset:offset + per_page]
        total_pages = (len(filtered) + per_page - 1) // per_page

        full_df = pd.read_csv(RESULTS_CSV)
        stats = {
            'total_rules': total,
            'avg_confidence': round(full_df['confidence'].mean() * 100, 1),
            'avg_lift': round(full_df['lift'].mean(), 2),
            'max_lift': round(full_df['lift'].max(), 2),
            'avg_support': round(full_df['support'].mean() * 100, 2),
        }

        chart_bar, chart_scatter, chart_network = '', '', ''
        if not shown_df.empty:
            chart_bar, chart_scatter = create_charts(shown_df)
            chart_network = create_network_chart(full_df)

        # Current filter values to re-populate form fields
        filters = {
            'min_lift': min_lift if min_lift > 0 else '',
            'max_lift': max_lift if max_lift < 9999 else '',
            'min_conf': min_conf if min_conf > 0 else '',
            'max_conf': max_conf if max_conf < 1 else '',
            'min_sup': min_sup if min_sup > 0 else '',
        }

        rerun_error = request.args.get('error', '')

        return render_template(
            'results.html',
            records=shown_df.to_dict('records'),
            chart_bar=chart_bar,
            chart_scatter=chart_scatter,
            chart_network=chart_network,
            total=total,
            shown=len(shown_df),
            page=page,
            total_pages=total_pages,
            query=query,
            stats=stats,
            filters=filters,
            dataset_stats=session.get('dataset_stats'),
            last_support=session.get('last_support', ''),
            last_confidence=session.get('last_confidence', ''),
            has_file=bool(session.get('last_file')),
            rerun_error=rerun_error,
        )
    except Exception as e:
        return f"An error occurred: {str(e)}"


@app.route('/download')
def download():
    df = pd.read_csv(RESULTS_CSV)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='AprioriResults.xlsx'
    )


@app.route('/download/csv')
def download_csv():
    df = pd.read_csv(RESULTS_CSV)
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='AprioriResults.csv'
    )


@app.route('/download/json')
def download_json():
    df = pd.read_csv(RESULTS_CSV)
    output = df.to_json(orient='records', indent=2)
    return send_file(
        BytesIO(output.encode()),
        mimetype='application/json',
        as_attachment=True,
        download_name='AprioriResults.json'
    )


if __name__ == '__main__':
    app.run(debug=True)
