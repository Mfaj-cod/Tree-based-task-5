import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier


def train_evaluate(x_train, x_test, y_train, y_test, feature_names=None, class_names=None):
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    reports_dir = os.path.join(os.getcwd(), 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            rocauc = roc_auc_score(y_test, y_pred)
        except ValueError:
            rocauc = "N/A (binary labels not suitable)"
        cm = confusion_matrix(y_test, y_pred)

        # save PDF report
        pdf_path = os.path.join(reports_dir, f"{model_name.replace(' ', '_')}_report.pdf")

        with PdfPages(pdf_path) as pdf:
            # Page 1
            fig = plt.figure(figsize=(8.27, 11.69))  # A4
            txt = (
                f"Model: {model_name}\n\n"
                f"Parameters: {model.get_params()}\n\n"
                f"Accuracy: {acc:.4f}\n"
                f"Precision: {prec:.4f}\n"
                f"Recall: {rec:.4f}\n"
                f"F1 Score: {f1:.4f}\n"
                f"ROC-AUC: {rocauc}\n\n"
                f"Confusion Matrix:\n{cm}\n\n"
                f"Classification Report:\n{classification_report(y_test, y_pred)}\n\n"
                f"Feature Importance:\n {model.feature_importances_}"
            )

            plt.text(
                0.05, 0.95, txt,
                transform=plt.gca().transAxes,
                fontsize=9,
                verticalalignment='top',
                family='monospace'
            )
            plt.title(f"{model_name} - Performance Report")

            pdf.savefig(fig)
            plt.close(fig)

            # Decision tree visualization
            if model_name == "Decision Tree":
                fig, ax = plt.subplots(figsize=(12, 8))
                plot_tree(
                    model,
                    filled=True,
                    feature_names=feature_names,
                    class_names=class_names,
                    rounded=True,
                    fontsize=8,
                    ax=ax
                )
                plt.title("Decision Tree Visualization", fontsize=12)
                pdf.savefig(fig)
                plt.close(fig)

        print(f"Report saved successfully at: {pdf_path}\n")
