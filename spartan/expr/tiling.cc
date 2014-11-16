#include <Python.h>
#include <unordered_map>
#include <list>

const int eMax = 1000000;
const int nMax = 5000;
const int INF = 1000000000;

struct Edge {
    int u, v, next;
    long cost;
} edge[eMax];

std::unordered_map<int, int> split_nodes;
int e, head[nMax];
long mincost, dis[nMax];
bool vis[nMax];

void add_edge(int u, int v, long cost) {
	edge[e].u = u; edge[e].v = v; edge[e].cost = cost;
    edge[e].next = head[u]; head[u] = e++;
}

void print_choices(int s, int t, bool* vis, long mincost) {
	printf("%d choose (cost=%ld):", s, mincost);
	for (int u = 0; u <= t; u++)
		if (vis[u]) printf("%d ", u);
	printf("\n");
}

long find_mincost_tiling(int s, int t, bool* vis) {
	long mincost = 0;
    int i, j, v, sp_v, size;
	std::list<int> child_edges;
	for (i = head[s]; i != -1; i = edge[i].next) child_edges.push_back(i);
	size = child_edges.size();

	while (!child_edges.empty()) {
		i = child_edges.front(); child_edges.pop_front(); size --;
		v = edge[i].v;

		if (split_nodes.find(v) != split_nodes.end()) { // splited node
			sp_v = split_nodes[v];

			// find split edge j
			for (j = edge[i].next; j != -1 && edge[j].v != sp_v; j=edge[j].next);

			if (j < 0) {   // not a two-edges-choose-one case
				if (vis[v] or vis[sp_v])
					mincost += (vis[v])? edge[i].cost : INF;
				else {
					dis[v] = find_mincost_tiling(v, t, vis);
					mincost += dis[v] + edge[i].cost;
					vis[v] = true;
				}
			} else {      // two edges we can only choose one
				child_edges.remove(j); size --;
				if (vis[v] or vis[sp_v])
					mincost += (vis[v])? edge[i].cost : edge[j].cost;
				else {
					bool vis1[nMax], vis2[nMax];
					memcpy(vis1, vis, t * sizeof(bool));
					memcpy(vis2, vis, t * sizeof(bool));
					dis[v] = find_mincost_tiling(v, t, vis1);
					dis[sp_v] = find_mincost_tiling(sp_v, t, vis2);
					long cmp = dis[v] + edge[i].cost - dis[sp_v] - edge[j].cost;
                    if (cmp == 0 && size > 0) {
						child_edges.push_back(i);
						child_edges.push_back(j);
					} else if (cmp < 0) {
						mincost += dis[v] + edge[i].cost;
						memcpy(vis, vis1, t * sizeof(bool));
						vis[v] = true;
					} else {
						mincost += dis[sp_v] + edge[j].cost;
						memcpy(vis, vis2, t * sizeof(bool));
						vis[sp_v] = true;
					}
				}
			}
		} else { // all must be chosen case
			if (vis[v]) mincost += edge[i].cost;
			else {
				dis[v] = find_mincost_tiling(v, t, vis);
				mincost += dis[v] + edge[i].cost;
				vis[v] = true;
			}
		}
	}
	//print_choices(s, t, vis, mincost);
	return mincost;
}

static PyObject* mincost_tiling(PyObject *self, PyObject *args) {
	PyObject *list, *ans;
	Py_ssize_t pos;
	int t, u, v;
    long cost;

	// init t node
	t = (int)PyInt_AsLong(PyTuple_GetItem(args, 0));

	// init edges
	e = 0;
	memset(head, -1, sizeof(head));
	list = PyTuple_GetItem(args, 1);
	for (pos = 0; pos < PyList_Size(list); pos++) {
		PyArg_ParseTuple(PyList_GetItem(list, pos), "iik", &u, &v, &cost);
		add_edge(u, v, cost);
		//printf("add edge:(%d, %d, cost=%ld)\n", u, v, cost);
	}

	// init splited nodes
	split_nodes.clear();
	list = PyTuple_GetItem(args, 2);
	for (pos = 0; pos < PyList_Size(list); pos++) {
		PyArg_ParseTuple(PyList_GetItem(list, pos), "ii", &u, &v);
		split_nodes[u] = v;
		split_nodes[v] = u;
		//printf("add split nodes:(%d, %d)\n", u, v);
	}

	memset(vis, false, sizeof(vis));
	mincost = find_mincost_tiling(0, t, vis);

	ans = PyList_New(0);
	for (u = 0; u < t; u++)
		if (vis[u]) PyList_Append(ans, Py_BuildValue("i", u));

	//printf("mincost:%ld\n", mincost);
	return ans;
}

static PyMethodDef TilingMethods[] = {
	{"mincost_tiling", mincost_tiling, METH_VARARGS, NULL},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC inittiling(void) {
	PyObject *m;
	m = Py_InitModule("tiling", TilingMethods);
	if (m == NULL) return;
}

