# script para usar mlflow tracking
import argparse
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             roc_auc_score,
                             balanced_accuracy_score,
                             f1_score)
from sklearn.model_selection import train_test_split

# %%
titanic_df = (pd.read_csv('https://raw.githubusercontent.com/edroga/Datasets_for_projects/main/titanic.csv')
               .drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
               .dropna()
               # usamos pipe para obtener summy variables para variables de Sex, Embarked y Pclass
               .pipe(lambda df_: pd.get_dummies(df_, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True))
               )

# %%
# Setup de los argumentos tomados de la linea de comando
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type = float, default = 0.1)
parser.add_argument('--penalty', type = str, default = 'l2')
args = parser.parse_args()

# %%
# Partimos los datos en train_test
X_train, X_test, y_train, y_test = train_test_split(
    titanic_df.drop(columns = 'Survived'),
    titanic_df['Survived'],
    test_size = 0.2,
    random_state = 123
)

# %%
# Empezar a usar Mlflow con el context manager
with mlflow.start_run(run_name = 'experimento_titanic'):

    # nombre del experimento
    mlflow.set_experiment('titanic experimento Jair')

    # loggear los parametros del modelo
    mlflow.log_param('alpha', args.alpha)
    mlflow.log_param('penalty', args.penalty)

    # instanciamos la regresion logistica
    model = LogisticRegression(C = args.alpha,
                               penalty = args.penalty,
                               random_state = 123,
                               max_iter = 1000)

    # Ajustar el modelo
    model.fit(X_train, y_train)

    # pronosticamos sobre el test
    y_pred = model.predict(X_test)

    # calculamos las metricas de evaluacion
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # loggear las metricas
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('roc_auc', roc_auc)
    mlflow.log_metric('balanced_accuracy', balanced_accuracy)
    mlflow.log_metric('f1', f1)


