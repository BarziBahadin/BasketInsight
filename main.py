import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class AprioriGUI:
    def __init__(self, root):
        self.root = root
        root.title('Apriori Algorithm GUI')
        root.option_add('*Font', 'Helvetica 12')
        self.csv_file_path = None
        self.rules = None
        self.frequent_itemsets = None
        self.df_encoded = None

        root.grid_rowconfigure(6, weight=1)
        root.grid_columnconfigure(1, weight=1)

        tk.Label(root, text='Min Support (%):').grid(row=0, column=0, sticky='w')
        self.min_support_entry = tk.Entry(root)
        self.min_support_entry.grid(row=0, column=1, sticky='ew')
        self.min_support_entry.insert(0, "5")

        tk.Label(root, text='Min Confidence (%):').grid(row=1, column=0, sticky='w')
        self.min_confidence_entry = tk.Entry(root)
        self.min_confidence_entry.grid(row=1, column=1, sticky='ew')
        self.min_confidence_entry.insert(0, "2")

        self.load_button = tk.Button(root, text='Load CSV', command=self.load_csv)
        self.load_button.grid(row=2, column=0, columnspan=2, sticky='ew')

        self.run_button = tk.Button(root, text='Run Apriori', command=self.run_apriori, state=tk.DISABLED)
        self.run_button.grid(row=3, column=0, columnspan=2, sticky='ew')

        # Search
        self.search_entry = tk.Entry(root)
        self.search_entry.grid(row=4, column=0, sticky='ew')
        self.search_entry.config(state=tk.DISABLED)
        self.search_button = tk.Button(root, text='Search', command=self.search_items)
        self.search_button.grid(row=4, column=1, sticky='ew')
        self.search_button.config(state=tk.DISABLED)

        # Results treeview frame
        self.results_frame = ttk.Frame(root)
        self.results_frame.grid(row=5, column=0, columnspan=2, sticky='nsew')
        root.grid_rowconfigure(5, weight=1)
        root.grid_columnconfigure(0, weight=1)
        self.tree = None

        self.export_button = tk.Button(root, text='Export to Excel', command=self.export_to_excel,
                                       bg='green', fg='black')
        self.export_button.grid(row=6, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        self.export_button.config(state=tk.DISABLED)

        self.run_kmeans_button = tk.Button(root, text='Run K-Means', command=self.run_kmeans,
                                           bg='blue', fg='white')
        self.run_kmeans_button.grid(row=7, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        self.run_kmeans_button.config(state=tk.DISABLED)

        self.report_button = tk.Button(root, text='Generate Report', command=self.generate_report,
                                       bg='orange', fg='black')
        self.report_button.grid(row=8, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        self.report_button.config(state=tk.DISABLED)

    def setup_results_treeview(self, columns):
        if self.tree is not None:
            self.tree.destroy()

        self.tree = ttk.Treeview(self.results_frame, columns=columns, show="headings")
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=120)

        self.tree.tag_configure('odd', background='#E8E8E8')
        self.tree.tag_configure('even', background='white')

        self.tree.grid(row=0, column=0, sticky='nsew')
        self.results_frame.grid_rowconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)

        vsb = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.tree.yview)
        vsb.grid(row=0, column=1, sticky='ns')
        self.tree.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(self.results_frame, orient="horizontal", command=self.tree.xview)
        hsb.grid(row=1, column=0, sticky='ew')
        self.tree.configure(xscrollcommand=hsb.set)

    def insert_data_to_tree(self, data_rows):
        for index, row in enumerate(data_rows):
            tag = 'even' if index % 2 == 0 else 'odd'
            self.tree.insert("", tk.END, values=row, tags=(tag,))

    def load_csv(self):
        filename = filedialog.askopenfilename(
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filename:
            self.csv_file_path = filename
            self.transactions = self.read_transactions(filename)
            messagebox.showinfo("Info", f"Loaded {len(self.transactions)} transactions.")
            self.run_button.config(state=tk.NORMAL)
            self.run_kmeans_button.config(state=tk.NORMAL)

    def run_apriori(self):
        try:
            min_support = float(self.min_support_entry.get()) / 100
            min_confidence = float(self.min_confidence_entry.get()) / 100
        except ValueError:
            messagebox.showerror("Error", "Min Support and Min Confidence must be numbers.")
            return

        try:
            te = TransactionEncoder()
            te_ary = te.fit(self.transactions).transform(self.transactions)
            df = pd.DataFrame(te_ary, columns=te.columns_)

            # Cache encoded DataFrame for K-Means reuse
            self.df_encoded = df.astype(int)

            self.frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

            if self.frequent_itemsets.empty:
                messagebox.showinfo("No Results", "No frequent itemsets found. Try lowering min support.")
                return

            self.rules = association_rules(
                self.frequent_itemsets, metric="confidence", min_threshold=min_confidence
            )

            if self.rules.empty:
                messagebox.showinfo("No Results", "No rules found. Try lowering min confidence.")
                return

            self.display_results(self.rules)
            self.export_button.config(state=tk.NORMAL)
            self.search_button.config(state=tk.NORMAL)
            self.search_entry.config(state=tk.NORMAL)
            self.report_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_results(self, rules):
        columns = ['#', 'Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift']
        self.setup_results_treeview(columns)

        for index, rule in enumerate(rules.itertuples(index=False), start=1):
            tag = 'even' if index % 2 == 0 else 'odd'
            self.tree.insert("", tk.END, values=(
                index,
                ', '.join(sorted(rule.antecedents)),
                ', '.join(sorted(rule.consequents)),
                round(rule.support, 4),
                round(rule.confidence, 4),
                round(rule.lift, 4)
            ), tags=(tag,))

    def search_items(self):
        if self.rules is None:
            return
        search_query = self.search_entry.get().lower()
        matching_items = []

        for _, row in self.rules.iterrows():
            antecedents = ', '.join(sorted(row['antecedents']))
            consequents = ', '.join(sorted(row['consequents']))
            if search_query in antecedents.lower() or search_query in consequents.lower():
                matching_items.append((
                    len(matching_items) + 1,
                    antecedents,
                    consequents,
                    round(row['support'], 4),
                    round(row['confidence'], 4),
                    round(row['lift'], 4)
                ))

        self.tree.delete(*self.tree.get_children())
        for index, item in enumerate(matching_items):
            tag = 'even' if index % 2 == 0 else 'odd'
            self.tree.insert("", tk.END, values=item, tags=(tag,))

        if not matching_items:
            messagebox.showinfo("Search", f"No rules found matching '{search_query}'.")

    def preprocess_data_for_kmeans(self, df):
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        df_filled = imputer.fit_transform(df)
        scaler = StandardScaler()
        return scaler.fit_transform(df_filled)

    def read_transactions(self, filename):
        transactions = []
        with open(filename, 'r', encoding='utf-8', errors='replace') as file:
            for line in file:
                items = [item.strip() for item in line.strip().split(',') if item.strip()]
                if items:
                    transactions.append(items)
        return transactions

    def run_kmeans(self):
        if not self.csv_file_path:
            messagebox.showerror("Error", "No CSV file loaded.")
            return

        # Ask for number of clusters first
        num_clusters = simpledialog.askinteger(
            "K-Means Clustering", "Number of clusters:",
            parent=self.root, minvalue=2, maxvalue=20
        )
        if num_clusters is None:
            return

        try:
            # Use cached one-hot encoded data if Apriori was already run
            if self.df_encoded is not None:
                data_matrix = self.df_encoded
            else:
                # Encode on the fly
                te = TransactionEncoder()
                te_ary = te.fit(self.transactions).transform(self.transactions)
                data_matrix = pd.DataFrame(te_ary, columns=te.columns_).astype(int)
                self.df_encoded = data_matrix

            data_scaled = self.preprocess_data_for_kmeans(data_matrix)

            if np.isnan(data_scaled).any() or np.isinf(data_scaled).any():
                messagebox.showerror("Error", "Scaled data contains NaN or infinite values.")
                return

            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            kmeans.fit(data_scaled)

            result_df = data_matrix.copy()
            result_df['Cluster'] = kmeans.labels_

            self.setup_results_treeview(list(result_df.columns))
            self.update_treeview_with_clusters(result_df)
            self.export_button.config(state=tk.NORMAL)
            messagebox.showinfo("K-Means", f"Clustering complete. {num_clusters} clusters assigned.")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_treeview_with_clusters(self, df):
        self.tree.delete(*self.tree.get_children())
        for index, row in enumerate(df.itertuples(index=False)):
            tag = 'even' if index % 2 == 0 else 'odd'
            self.tree.insert("", tk.END, values=list(row), tags=(tag,))

    def generate_report(self):
        if self.frequent_itemsets is None or self.frequent_itemsets.empty:
            messagebox.showerror("Error", "No analysis data available. Run Apriori first.")
            return

        try:
            fast_threshold = self.frequent_itemsets['support'].quantile(0.75)
            slow_threshold = self.frequent_itemsets['support'].quantile(0.25)

            fast_items = self.frequent_itemsets[self.frequent_itemsets['support'] >= fast_threshold]
            slow_items = self.frequent_itemsets[self.frequent_itemsets['support'] <= slow_threshold]

            report_lines = [
                "=== MARKET BASKET ANALYSIS REPORT ===\n",
                f"Total frequent itemsets: {len(self.frequent_itemsets)}",
                f"Total rules generated:   {len(self.rules) if self.rules is not None else 'N/A'}",
                "",
                f"--- Fast-Selling Items (top 25%, support >= {fast_threshold:.4f}) ---",
                fast_items.to_string(index=False),
                "",
                f"--- Slow-Selling Items (bottom 25%, support <= {slow_threshold:.4f}) ---",
                slow_items.to_string(index=False),
            ]
            report_text = "\n".join(report_lines)

            save_path = filedialog.asksaveasfilename(
                defaultextension='.txt',
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Report"
            )
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                messagebox.showinfo("Report Saved", f"Report saved to:\n{save_path}")
            else:
                # Show in dialog if user cancelled save
                messagebox.showinfo("Sales Report", report_text[:2000])

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export_to_excel(self):
        if not self.tree.get_children():
            messagebox.showerror("Error", "No data to export.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if not filename:
            return

        columns = [self.tree.heading(col)['text'] for col in self.tree['columns']]
        data_rows = [self.tree.item(item)['values'] for item in self.tree.get_children()]
        df = pd.DataFrame(data_rows, columns=columns)

        try:
            df.to_excel(filename, index=False)
            messagebox.showinfo("Success", "Data exported successfully to Excel.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export to Excel.\n{e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = AprioriGUI(root)
    root.minsize(600, 400)
    root.mainloop()
