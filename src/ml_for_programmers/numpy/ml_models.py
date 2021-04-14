# %%
from IPython import get_ipython

# %% [markdown]
# <img src="img/python-logo-notext.svg"
#      style="display:block;margin:auto;width:10%"/>
# <h1 style="text-align:center;">Python: NumPy</h1>
# <h2 style="text-align:center;">Coding Akademie München GmbH</h2>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# %% [markdown]
# # Listen als Vektoren und Matrizen
# 
# Wir können Python Listen verwenden um Vektoren darzustellen:

# %%
vector1 = [3, 2, 4]
vector2 = [8, 9, 7]

# %% [markdown]
# Es wäre dann möglich, Vektoroperationen auf derartigen Listen zu implementieren:

# %%
def vector_sum(v1, v2):
    assert len(v1) == len(v2)
    result = [0] * len(v1)
    for i in range(len(v1)):
        result[i] = v1[i] + v2[i]
    return result


# %%
vector_sum(vector1, vector2)

# %% [markdown]
# Matrizen könnten dann als "Listen von Listen" dargestellt werden:

# %%
matrix = [[1, 2, 3],
          [2, 3, 4],
          [3, 4, 5]]

# %% [markdown]
# Diese Implementierungsvariante hat jedoch einige Nachteile:
# - Performanz
#     - Speicher
#     - Geschwindigkeit
#     - Parallelisierbarkeit
# - Interface
#     - Zu allgemein
#     - `*`, `+` auf Listen entspricht nicht den Erwartungen
#     - ...
# - ...
# %% [markdown]
# # NumPy
# 
# NumPy ist eine Bibliothek, die einen Datentyp für $n$-dimensionale Tensoren (`ndarray`) sowie effiziente Operationen darauf bereitstellt.
# - Vektoren
# - Matrizen
# - Grundoperationen für Lineare Algebra
# - Tensoren für Deep Learning
# 
# Fast alle anderen mathematischen und Data-Science-orientierten Bibliotheken für Python bauen auf NumPy auf (Pandas, SciPy, Statsmodels, TensorFlow, ...).
# %% [markdown]
# ## Überblick

# %%
import numpy as np


# %%
v1 = np.array([3, 2, 4])
v2 = np.array([8, 9, 7])


# %%
type(v1)


# %%
v1.dtype


# %%
v1 + v2


# %%
v1 * v2 # Elementweises (Hadamard) Produkt


# %%
v1.dot(v2)


# %%
v1.sum()


# %%
v1.mean()


# %%
v1.max()


# %%
v1.argmax(), v1[v1.argmax()]


# %%
m1 = np.array([[1, 2, 3],
               [4, 5, 6]])
m2 = np.array([[1, 0],
               [0, 1],
               [2, 3]])


# %%
# m1 + m2


# %%
m1.T


# %%
m1.T + m2


# %%
m1.dot(m2)

# %% [markdown]
# ## Erzeugen von NumPy Arrays
# 
# ### Aus Python Listen
# 
# Durch geschachtelte Listen lassen sich Vektoren, Matrizen und Tensoren erzeugen:

# %%
vector = np.array([1, 2, 3, 4])
vector


# %%
vector.shape


# %%
matrix = np.array([[1, 2, 3], [4, 5, 6]])
matrix


# %%
matrix.shape


# %%
tensor = np.array([[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]]])
tensor


# %%
tensor.shape

# %% [markdown]
# ### Als Intervall bzw. Folge

# %%
np.arange(10)


# %%
np.arange(10.0)


# %%
np.arange(2, 10)


# %%
np.arange(3., 23., 5.)


# %%
np.linspace(0, 10, 5)


# %%
np.linspace(0.1, 1, 10)


# %%
np.arange(0.1, 1.1, 0.1)

# %% [markdown]
# ### Konstant 0 oder 1

# %%
np.zeros(3)


# %%
np.zeros((3,))


# %%
np.zeros((3, 3))


# %%
np.ones(2)


# %%
np.ones((4, 5))

# %% [markdown]
# ### Als Identitätsmatrix

# %%
np.eye(4)

# %% [markdown]
# ### Aus Zufallsverteilung
# 
# Numpy bietet eine große Anzahl von möglichen [Generatoren und Verteilungen](https://docs.scipy.org/doc/numpy/reference/random/index.html) zum Erzeugen von Vektoren und Arrays mit zufälligen Elementen.
# %% [markdown]
# #### Setzen des Seed-Wertes

# %%
np.random.seed(101)

# %% [markdown]
# #### Gleichverteilt in [0, 1)

# %%
# Kompatibilität mit Matlab
np.random.seed(101)
np.random.rand(10)


# %%
np.random.rand(4, 5)


# %%
# Fehler
# np.random.rand((4, 5))


# %%
np.random.seed(101)
np.random.random(10)


# %%
np.random.random((4, 5))

# %% [markdown]
# #### Normalverteilte Zufallszahlen

# %%
# Kompatibilität mit Matlab
np.random.seed(101)
np.random.randn(10)


# %%
np.random.randn(4, 5)


# %%
# Fehler
# np.random.randn((4, 5))


# %%
np.random.seed(101)
np.random.standard_normal(10)


# %%
np.random.standard_normal((4, 5))


# %%
np.random.seed(101)
np.random.normal(10.0, 1.0, 10)


# %%
np.random.normal(0.0, 1.0, (4, 5))


# %%
np.random.normal(10.0, 0.2, (2, 5))

# %% [markdown]
# #### Multivariate Normalverteilung
# 

# %%
means = np.array([0.0, 2.0, 1.0])
cov = np.array([[2.0, -1.0, 0.0],
                [-1.0, 2.0, -1.0],
                [0.0, -1.0, 2.0]])
np.random.multivariate_normal(means, cov, (3,))

# %% [markdown]
# #### Andere Verteilungen

# %%
np.random.binomial(10, 0.2, 88)


# %%
np.random.multinomial(20, [1/6.0] * 6, 10)

# %% [markdown]
# Die [Dokumentation](https://docs.scipy.org/doc/numpy/reference/random/generator.html) enthält eine Liste aller Verteilungen und ihrer Parameter.
# %% [markdown]
# ## Mini-Workshop
# 
# - Notebook `050x-Workshop NumPy`
# - Abschnitt "Erzeugen von NumPy Arrays"
# 
# %% [markdown]
# ## Exkurs: Lösen von Gleichungssystemen
# 
# Wie können wir das folgende Gleichungssystem mit NumPy darstellen und lösen:
# 
# $$
# 2x_0 + x_1 + x_2 = 4\\
# x_0 + 3x_1 + 2x_2 = 5\\
# x_0 = 6
# $$

# %%
a = np.array([[2., 1., 1.],
              [1., 3., 2.],
              [1., 0., 0.]])
b = np.array([4., 5., 6.])


# %%
x = np.linalg.solve(a, b)
x


# %%
# Test:
a.dot(x), b

# %% [markdown]
# SciPy bietet spezielle Lösungsverfahren wie LU-Faktorisierung, Cholesky-Faktorisierung, etc. an.

# %%
import scipy.linalg as linalg
lu = linalg.lu_factor(a)


# %%
lu


# %%
x = linalg.lu_solve(lu, b)


# %%
x


# %%
a.dot(x)


# %%
# Hermite'sche Matrix, positiv definit
a = np.array([[10., -1., 2., 0.],
             [-1., 11., -1., 3.],
             [2., -1., 10., -1.],
             [0., 3., -1., 8.]])
b= np.array([6., 25., -11., 15.])


# %%
cholesky = linalg.cholesky(a)


# %%
cholesky


# %%
cholesky.T.conj().dot(cholesky)


# %%
y = np.linalg.solve(cholesky.T.conj(), b)


# %%
x = np.linalg.solve(cholesky, y)


# %%
x


# %%
a.dot(x)

# %% [markdown]
# ## Mini-Workshop
# 
# - Notebook `050x-Workshop NumPy`
# - Abschnitt "Gleichungssysteme"
# %% [markdown]
# ## Attribute von Arrays

# %%
int_array = np.arange(36)
float_array = np.arange(36.0)


# %%
int_array.dtype


# %%
float_array.dtype


# %%
int_array.shape


# %%
int_array.size


# %%
int_array.itemsize


# %%
float_array.itemsize


# %%
np.info(int_array)


# %%
np.info(float_array)

# %% [markdown]
# ## Ändern von Shape und Größe

# %%
float_array.shape


# %%
float_matrix = float_array.reshape((6, 6))


# %%
float_matrix


# %%
float_matrix.shape


# %%
float_array.shape


# %%
float_array.reshape(3, 12)


# %%
# Fehler
# float_array.reshape(4, 8)


# %%
float_array.reshape((4, 9), order='F')


# %%
float_array.reshape((9, 4)).T


# %%
np.resize(float_array, (4, 8))


# %%
float_array.shape


# %%
np.resize(float_array, (8, 10))

# %% [markdown]
# ## Mini-Workshop
# 
# - Notebook `050x-NumPy`
# - Abschnitt "Erzeugen von NumPy Arrays 2"
# 
# %% [markdown]
# ## Broadcasting von Operationen
# 
# Viele Operationen mit Skalaren werden Elementweise auf NumPy Arrays angewendet:

# %%
arr = np.arange(8)
arr


# %%
arr + 5


# %%
arr * 2


# %%
arr ** 2


# %%
2 ** arr


# %%
arr > 5

# %% [markdown]
# # Exkurs: K-Nearest Neighbors
# 
# K-Nearest Neighbors ist ein nichtparametrisches ML-Verahren mit dem man Regression und Klassifikation durchführen kann.
# 
# ## Grundidee (Klassifikation)
# 
# - Speichere alle Trainingsdaten $X_i$ und ihre Labels $y_i$
# - (Es gibt kein Training)
# - Um einen Wert $X_t$ zu klassifizieren suche die $k$ Werte aus den gespeicherten $X_i$, die die geringste Distanz zu $X_t$ haben und wähle das Label, das am häufigsten vorkommt
# 
# Bei der Regression mit KNN wird statt dem häufigsten Label ein (möglicherweise gewichteter) Mittelwert aus den Werten der $k$ Nachbarn gebildet.
# %% [markdown]
# ## Beispiel: Regression mit KNN
# 
# Im folgenden Beispiel verwenden wir KNN um Werte zwischen zufälligen Sapmles einer Funktion zu interpolieren. Das ist eine klassische Regressionsaufgabe.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


# %%
np.random.seed(12)
NUM_SAMPLES = 150
MAX_X = 8
RANDOM_SCALE=0.5


# %%
X = np.sort(MAX_X * np.random.random(NUM_SAMPLES))
X = X.reshape(-1, 1)
X[:3]


# %%
plt.figure(figsize=(20, 1), frameon=False)
plt.yticks([], [])
plt.scatter(X, np.zeros(X.shape), alpha=0.4);


# %%
def fun(x):
    return 2 * np.sin(x) + 0.1 * x ** 2 - 2


# %%
Xs = np.linspace(0, MAX_X, 500).reshape(-1, 1)
plt.figure(figsize=(20, 5))
plt.plot(Xs, fun(Xs));


# %%
y = fun(X) + np.random.normal(size=X.shape, scale=RANDOM_SCALE)
y = y.reshape(-1)


# %%
plt.figure(figsize=(20, 5))
plt.plot(Xs, fun(Xs));
plt.scatter(X, y, color='orange');


# %%
n_neighbors = 25
knn = KNeighborsRegressor(n_neighbors, weights='uniform')
knn.fit(X, y)


# %%
ys = knn.predict(Xs)


# %%
true_ys = fun(Xs)


# %%
mean_squared_error(true_ys, ys), mean_squared_error(y, knn.predict(X))


# %%
def plot_prediction(ys):
    plt.figure(figsize=(15,6))
    plt.scatter(X, y, color='orange', label='samples')
    plt.plot(Xs, ys, color='blue', label='predictions')
    plt.plot(Xs, true_ys, color='goldenrod', label='true_values')
    plt.legend()
plot_prediction(ys);


# %%
knn_dist = KNeighborsRegressor(n_neighbors, weights='distance')
knn_dist.fit(X, y)


# %%
ys_dist = knn_dist.predict(Xs)


# %%
mean_squared_error(true_ys, ys_dist), mean_squared_error(y, knn_dist.predict(X))


# %%
plot_prediction(ys_dist);

# %% [markdown]
# # Speichern von SK-Learn Modellen
# 
# SK-Learn verwendet das in Python standardmäßig vorhandene `pickle`-Modul um Modelle zu speichern und zu laden:

# %%
import pickle

with open('sklearn-knn.pickle', 'wb') as file:
    pickle.dump(knn, file)


# %%
with open('sklearn-knn.pickle', 'rb') as file:
    knn_loaded = pickle.load(file)


# %%
ys_loaded = knn.predict(Xs)
ys_loaded[:3]


# %%
mean_squared_error(true_ys, ys_loaded), mean_squared_error(y, knn_loaded.predict(X))


# %%
plot_prediction(ys_loaded);

# %% [markdown]
# ## Mini-Workshop
# 
# - Notebook `050x-NumPy`
# - Abschnitt "Regression mit KNN"
# 
# %% [markdown]
# # Ensembles
# 
# - Kombination mehrerer Estimators um bessere Performance zu erreichen

# %%
from sklearn.ensemble import BaggingRegressor
knn_ens = BaggingRegressor(KNeighborsRegressor(3), max_samples=0.2, max_features=1)
knn_ens.fit(X, y)


# %%
ys_ens = knn_ens.predict(Xs)


# %%
mean_squared_error(true_ys, ys_ens), mean_squared_error(y, knn_ens.predict(X))


# %%
plot_prediction(ys_ens);

# %% [markdown]
# ## Mini-Workshop
# 
# - Notebook `050x-NumPy`
# - Abschnitt "Regression mit KNN": Mit Ensembles.
# 
# %% [markdown]
# # Entscheidungsbäume
# 
# - Konstruktion von verschachtelten "if/then/else" Abfragen.

# %%
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()

dt.fit(X, y)

ys_dt = dt.predict(Xs)
mean_squared_error(true_ys, ys_dt), mean_squared_error(y, dt.predict(X))


# %%
plot_prediction(ys_dt);


# %%
dt3 = DecisionTreeRegressor(max_depth=3)
dt3.fit(X, y)

ys_dt3 = dt3.predict(Xs)

mean_squared_error(true_ys, ys_dt3), mean_squared_error(y, dt3.predict(X))


# %%
plot_prediction(ys_dt3);


# %%
from sklearn.tree import export_graphviz
export_graphviz(dt3, 'decision-tree3.dot')


# %%
dt4 = DecisionTreeRegressor(max_depth=4)
dt4.fit(X, y)

ys_dt4 = dt4.predict(Xs)

mean_squared_error(true_ys, ys_dt4), mean_squared_error(y, dt4.predict(X))


# %%
plot_prediction(ys_dt4);


# %%
from sklearn.tree import export_graphviz
export_graphviz(dt4, 'decision-tree4.dot')


# %%
from sklearn.tree import export_graphviz
export_graphviz(dt, 'decision-tree.dot')

# %% [markdown]
# # Ensembles von Entscheidungsbäumen
# 
# - Bei Entscheidungsbäumen gibt es mehr Möglichkeiten Ensembles zu erzeugen:
#     - Averaging: Bildung von Mittelwerten mehrerer starker Estimators
#     - Boosting: Bilden einer Sequenz von schwachen Estimators, bei der spätere Estimators gezielt die Schwächen der ersten beseitigen

# %%
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, max_depth=5)
rf.fit(X, y)

ys_rf = rf.predict(Xs)

mean_squared_error(true_ys, ys_rf), mean_squared_error(y, rf.predict(X))


# %%
plot_prediction(ys_rf);


# %%
from sklearn.ensemble import AdaBoostRegressor
ab = AdaBoostRegressor(n_estimators=100, base_estimator=DecisionTreeRegressor(max_depth=4))
ab.fit(X, y)

ys_ab = ab.predict(Xs)

mean_squared_error(true_ys, ys_ab), mean_squared_error(y, ab.predict(X))


# %%
plot_prediction(ys_ab);

# %% [markdown]
# ## Mini-Workshop
# 
# - Notebook `050x-NumPy`
# - Abschnitt "Regression mit KNN": Mit Entscheidungsbäumen und Ensembles von Entscheidungsbäumen.
# 
# %% [markdown]
# # Bilderkennung mit KNNs
# 
# Wir wollen den MNIST Datensatz mit Hilfe von KNNs klassifizieren. Um konsistent mit den Deep Learning Aufgaben zu sein verwenden wir den Datensatz aus `tensorflow.keras.datasets`.

# %%
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


# %%
(X_train_in, y_train), (X_test_in, y_test) = mnist.load_data()


# %%
def plot_random_datapoint():
    sample = np.random.randint(0, X_train_in.shape[0])

    plt.figure(figsize = (10,10))
    mnist_img = X_train_in[sample]
    plt.imshow(mnist_img, cmap="Greys")

    # Get the `Axes` instance on the current figure
    ax = plt.gca()

    plt.tick_params(
        axis='both', which='major', bottom=True, left=True,
        labelbottom=False, labelleft=False)

    plt.tick_params(
        axis='both', which='minor', bottom=False, left=False,
        labelbottom=True, labelleft=True)

    ax.set_xticks(np.arange(-.5, 28, 1))
    ax.set_yticks(np.arange(-.5, 28, 1))

    ax.set_xticks(np.arange(0, 28, 1), minor=True);
    ax.set_xticklabels([str(i) for i in np.arange(0, 28, 1)], minor=True);
    ax.set_yticks(np.arange(0, 28, 1), minor=True);
    ax.set_yticklabels([str(i) for i in np.arange(0, 28, 1)], minor=True);

    ax.grid(color='black', linestyle='-', linewidth=1.5)
    plt.colorbar(fraction=0.046, pad=0.04, ticks=[0,32,64,96,128,160,192,224,255])


# %%
plot_random_datapoint()


# %%
def preprocess_data(data):
    return data.reshape(-1, 28 * 28)


# %%
X_train = preprocess_data(X_train_in)
X_test = preprocess_data(X_test_in)


# %%
def shuffle(X, y):
    Xs = np.column_stack((X, y))
    np.random.shuffle(Xs)
    return Xs[:, :-1], Xs[:, -1]


# %%
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)


# %%
X_train, y_train = X_train[:5000], y_train[:5000]
X_test, y_test = X_test[:1000], y_test[:1000]


# %%
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# %%
n_neighbors = 5
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)


# %%
y_pred = knn.predict(X_test)
y_pred[:10], y_test[:10]


# %%
(accuracy_score(y_test, y_pred),
 precision_score(y_test, y_pred, average='macro'),
 recall_score(y_test, y_pred, average='macro'))

# %% [markdown]
# ## Mini-Workshop
# 
# - Notebook `050x-NumPy`
# - Abschnitt "Bilderkennung mit KNNs"
# 
# %% [markdown]
# ## Mini-Workshop
# 
# - Notebook `050x-NumPy`
# - Abschnitt "Bilderkennung mit Ensembles und Entscheidungsbäumen"
# 
# %% [markdown]
# ## Minimum, Maximum, Summe, ...

# %%
np.random.seed(101)
vec = np.random.rand(10)
vec


# %%
vec.max()


# %%
vec.argmax()


# %%
vec.min()


# %%
vec.argmin()


# %%
np.random.seed(101)
arr = np.random.rand(2, 5)
arr


# %%
arr.max()


# %%
arr.argmax()


# %%
arr.min()


# %%
arr.argmin()

# %% [markdown]
# ## Mini-Workshop
# 
# - Notebook `050x-NumPy`
# - Abschnitt "Extrema"
# 

# %%
arr.reshape(arr.size)[arr.argmin()]


# %%
arr[np.unravel_index(arr.argmin(), arr.shape)]


# %%
arr


# %%
arr.sum()


# %%
arr.sum(axis=0)


# %%
arr.sum(axis=1)


# %%
arr.mean()


# %%
arr.mean(axis=0)


# %%
arr.mean(axis=1)

# %% [markdown]
# ## Mini-Workshop
# 
# - Notebook `050x-NumPy`
# - Abschnitt "Mittelwert"
# 
# %% [markdown]
# ## Exkurs: Einfache Monte Carlo Simulation
# 
# Mit der folgenden Monte Carlo Simulation kann eine Approximation von $\pi$ berechnet werden.
# 
# Die Grundidee ist zu berechnen, welcher Anteil an zufällig gezogenen Paaren aus Zahlen $(x, y)$, mit $x, y \sim SV[0, 1)$  (d.h., unabhängig und stetig auf $[0, 1)$ verteilt) eine $\ell^2$ Norm kleiner als 1 hat. Diese Zahl ist eine
# Approximation von $\pi/4$.
# 
# Die folgende naive Implementiertung is in (fast) reinem Python geschrieben und verwendet NumPy nur zur Berechnung der Zufallszahlen.

# %%
def mc_pi_1(n):
    num_in_circle = 0
    for i in range(n):
        xy = np.random.random(2)
        if (xy ** 2).sum() < 1:
            num_in_circle += 1
    return num_in_circle * 4 / n


# %%
def test(mc_pi):
    np.random.seed(64)
    for n in [100, 10_000, 100_000, 1_000_000]:
        get_ipython().run_line_magic('time', 'print(f"𝜋 ≈ {mc_pi(n)} ({n} iterations).")')


# %%
test(mc_pi_1)

# %% [markdown]
# Durch Just-in-Time Übersetzung mit Numba kann die Performance erheblich gesteigert werden:

# %%
import numba
mc_pi_1_nb = numba.jit(mc_pi_1)


# %%
test(mc_pi_1_nb)

# %% [markdown]
# Die folgende Implementierung verwendet die Vektorisierungs-Features von NumPy:

# %%
def mc_pi_2(n):
    x = np.random.random(n)
    y = np.random.random(n)
    return ((x ** 2 + y ** 2) < 1).sum() * 4 / n


# %%
test(mc_pi_2)


# %%
# %time mc_pi_2(100_000_000)

# %% [markdown]
# Auch bei dieser Version können mit Numba Performance-Steigerungen erzielt werden, aber in deutlich geringerem Ausmaß:

# %%
mc_pi_2_nb = numba.jit(mc_pi_2)


# %%
test(mc_pi_2_nb)


# %%
# %time mc_pi_2_nb(100_000_000)

# %% [markdown]
# ## Mini-Workshop
# 
# - Notebook `050x-NumPy`
# - Abschnitt "Roulette"
# 
# %% [markdown]
# ## Indizieren von NumPy Arrays

# %%
vec = np.arange(10)


# %%
vec


# %%
vec[3]


# %%
vec[3:8]


# %%
vec[-1]


# %%
arr = np.arange(24).reshape(4, 6)


# %%
arr


# %%
arr[1]


# %%
arr[1][2]


# %%
arr[1, 2]


# %%
arr


# %%
arr[1:3]


# %%
arr[1:3][2:4]


# %%
arr[1:3, 2:4]


# %%
arr[:, 2:4]


# %%
# Vorsicht!
arr[: 2:4]


# %%
arr[:, 1:6:2]

# %% [markdown]
# ## Broadcasting auf Slices
# 
# In NumPy Arrays werden Operationen oftmals auf Elemente (oder Unterarrays) "gebroadcastet":

# %%
arr = np.ones((3, 3))


# %%
arr[1:, 1:] = 2.0


# %%
arr


# %%
lst = [1, 2, 3]
vec = np.array([1, 2, 3])


# %%
lst[:] = [99]


# %%
vec[:] = [99]


# %%
lst


# %%
vec


# %%
vec[:] = 11
vec

# %% [markdown]
# ### Vorsicht beim `lst[:]` Idiom! 

# %%
lst1 = list(range(10))
lst2 = lst1[:]
vec1 = np.arange(10)
vec2 = vec1[:]


# %%
lst1[:] = [22] * 10
lst1


# %%
lst2


# %%
vec1[:] = 22
vec1


# %%
vec2


# %%
vec1 = np.arange(10)
vec2 = vec1.copy()


# %%
vec1[:] = 22
vec1


# %%
vec2

# %% [markdown]
# ## Bedingte Selektion
# 
# NumPy Arrays können als Index auch ein NumPy Array von Boole'schen Werten erhalten, das den gleichen Shape hat wie das Array.
# 
# Dadurch werden die Elemente selektiert, an deren Position der Boole'sche Vektor den Wert `True` hat und als Vektor zurückgegeben.

# %%
vec = np.arange(8)
bool_vec = (vec % 2 == 0)


# %%
vec[bool_vec]


# %%
arr = np.arange(8).reshape(2, 4)
bool_arr = (arr % 2 == 0)
bool_arr


# %%
arr[bool_arr]


# %%
# Fehler!
# arr[bool_vec]


# %%
vec[vec % 2 > 0]


# %%
arr[arr < 5]

# %% [markdown]
# ### Boole'sche Operationen auf NumPy Arrays

# %%
bool_vec


# %%
neg_vec = np.logical_not(bool_vec)


# %%
bool_vec & neg_vec


# %%
bool_vec | neg_vec

# %% [markdown]
# ## Universelle NumPy Operationen
# 
# NumPy bietet viele "universelle" Funktionen an, die auf NumPy Arrays, Listen und Zahlen angewendet werden können:

# %%
vec1 = np.random.randn(5)
vec2 = np.random.randn(5)
list1 = list(vec1)
list2 = list(vec2)


# %%
vec1


# %%
list1


# %%
np.sin(vec1)


# %%
np.sin(list1)


# %%
import math
np.sin(math.pi)


# %%
np.sum(vec1)


# %%
np.sum(list1)


# %%
np.mean(vec1)


# %%
np.median(vec1)


# %%
np.std(vec1)


# %%
np.greater(vec1, vec2)


# %%
np.greater(list1, list2)


# %%
np.greater(vec1, list2)


# %%
np.maximum(vec1, vec2)


# %%
np.maximum(list1, list2)


# %%
np.maximum(list1, vec2)

# %% [markdown]
# Eine vollständige Liste sowie weitere Dokumentation findet man [hier](https://docs.scipy.org/doc/numpy/reference/ufuncs.html).

# %%



