import pydot
from sklearn.externals.six import StringIO
from sklearn import tree
from drift_detection import load_data

[X,Y] = load_data()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("sample.pdf")
