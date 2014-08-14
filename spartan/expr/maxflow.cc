#include <Python.h>
#include <queue>
#include <unordered_map>
#include <iostream>
using namespace std;

const int eMax = 1000000;
const int nMax = 1000;
const int INF = 1000000000;

struct Edge {
	int u, v, next, cap, cost;
} edge[eMax];
//unordered_map<int, int> split_edges;
unordered_map<int, int> split_nodes;
int maxflow, mincost, e;

int head[nMax], dis[nMax];
bool vis[nMax];

//void add_edge(int u, int v, int cap, int cost){
//	edge[e].u = u; edge[e].v = v; edge[e].cap = cap; edge[e].cost = cost;
//    edge[e].next = head[u]; head[u] = e++;
//
//    edge[e].u = v; edge[e].v = u; edge[e].cap = 0; edge[e].cost = -cost;
//    edge[e].next = head[v]; head[v] = e++;
//}

//bool spfa(int s, int t)
//{
//    int i, u, v;
//    queue <int> que;
//    for(i = 0; i <= t; i++) {
//    	dis[i] = INF;
//    	vis[i] = false;
//    	pre[i] = -1;
//    }
//
//    dis[s] = 0;
//    que.push(s);
//    vis[s] = true;
//    while (!que.empty()) {
//    	u = que.front(); que.pop(); vis[u] = false;
//    	for (i = head[u]; i!= -1; i = edge[i].next) {
//    		v = edge[i].v;
//    		cout << "try edge:(" << edge[i].u << "," << edge[i].v << ")" << endl;
//    		if (split_edges.find(i) != split_edges.end() && edge[split_edges[i]].u == edge[i].u &&
//    			edge[split_edges[i]^1].cap && split_edges[i] != (pre[edge[i].u]^1)) continue;
//
//    		int pe = pre[edge[i].u];
//    		if (split_edges.find(pe) != split_edges.end() && edge[split_edges[pe]].u != edge[pe].u &&
//    			edge[split_edges[pe]^1].cap && i != (split_edges[pe]^1)) continue;
//
//    		cout << "allow edge:(" << edge[i].u << "," << edge[i].v << ")" << endl;
//    		if (edge[i].cap && dis[v] > dis[u] + edge[i].cost) {
//    			dis[v] = dis[u] + edge[i].cost;
//    			pre[v] = i;
//    			if (!vis[v]) {
//    				que.push(v);
//    				vis[v] = true;
//    			}
//    		}
//    	}
//    }
//    if(dis[t] == INF) return false;
//    return true;
//}

//static PyObject* mincost_maxflow(PyObject *self, PyObject *args) {
//	PyObject *dict, *key, *value, *ans;
//	Py_ssize_t pos;
//	int t, u, v, ov, cap, cost;
//
//	t = (int)PyInt_AsLong(PyTuple_GetItem(args, 0));
//
//	// init edges
//	memset(head, -1, sizeof(head));
//	e = 0;
//	dict = PyTuple_GetItem(args, 1);
//	pos = 0;
//	while (PyDict_Next(dict, &pos, &key, &value)) {
//		PyArg_ParseTuple(key, "ii", &u, &v);
//		PyArg_ParseTuple(value, "ii", &cap, &cost);
//		add_edge(u, v, cap, cost);
//		//cout << "add edge:(" << u << " " << v << " " << cap << " " << cost << ")" << endl;
//	}
//
//	// init all splited edges
//	split_edges.clear();
//	dict = PyTuple_GetItem(args, 2);
//	pos = 0;
//	while (PyDict_Next(dict, &pos, &key, &value)) {
//		u = (int)PyInt_AsLong(key);
//		PyArg_ParseTuple(value, "ii", &v, &ov);
//		cout << "add split edge:(" << u << " " << v << " " << ov << ")" << endl;
//		if (u < 0) {
//			for (int i = head[v]; i != -1; i = edge[i].next)
//				if (edge[i].v == -u) v = i;
//			for (int i = head[ov]; i != -1; i = edge[i].next)
//				if (edge[i].v == -u) ov = i;
//		} else {
//			for (int i = head[u]; i != -1; i = edge[i].next) {
//				if (edge[i].v == v) v = i;
//				if (edge[i].v == ov) ov = i;
//			}
//		}
//		split_edges[v] = ov;
//		split_edges[ov] = v;
//	}
//
//	maxflow = 0;
//	mincost = 0;
//	ans = PySet_New(NULL);
//	while (spfa(0, t)) {
//		cap = INF;
//		for (u = pre[t]; u != -1; u = pre[edge[u].u]) {
//			cap = min(cap, edge[u].cap);
//		}
//		maxflow += cap;
//
//		cout << "road:";
//		for (u = pre[t]; u != -1; u = pre[edge[u].u]) {
//			edge[u].cap -= cap;
//			edge[u^1].cap += cap;
//			mincost += cap * edge[u].cost;
//			cout << '(' << edge[u].u << ',' << edge[u].v << ')';
//		}
//		cout << " " << mincost << endl;
//	}
//
//	for (u = 0; u < e; u += 2)
//		if (edge[u].cap == 0) {
//			PySet_Add(ans, Py_BuildValue("i", edge[u].v));
//			//cout << '(' << edge[u].u << ',' << edge[u].v << ')';
//		}
//	cout << endl;
//	cout << "maxflow:" << maxflow << " mincost:" << mincost << endl;
//	return ans;
//}

void add_edge(int u, int v, int cap, int cost){
	edge[e].u = u; edge[e].v = v; edge[e].cap = cap; edge[e].cost = cost;
    edge[e].next = head[u]; head[u] = e++;
}

void print_visited(bool* vis, int t) {
	for (int i = 0; i <= t; i++)
		if (vis[i]) cout << i << " ";
	cout << endl;
}

int find_mincost_flow(int s, int t, bool* vis) {
	int mincost = 0, i, j, v, sp_v = -1;

	for (i = head[s]; i != -1; i = edge[i].next) {
		v = edge[i].v;
		if (split_nodes.find(v) != split_nodes.end()) {
			if (v == sp_v) continue; // already calculated
			sp_v = split_nodes[v];

			for (j = edge[i].next; j != -1 && edge[j].v != sp_v; j=edge[j].next);

			if (j < 0) { // not a two-edges-choose-one case
				if (vis[v] or vis[sp_v])
					mincost += (vis[v])? edge[i].cost : INF;
				else {
					dis[v] = find_mincost_flow(v, t, vis);
					mincost += dis[v] + edge[i].cost;
					vis[v] = true;
				}
			} else { // two edges we can only choose one
				if (vis[v] or vis[sp_v])
					mincost += (vis[v])? edge[i].cost : edge[j].cost;
				else {
					bool vis1[nMax], vis2[nMax];
					memcpy(vis1, vis, t * sizeof(bool));
					memcpy(vis2, vis, t * sizeof(bool));
					dis[v] = find_mincost_flow(v, t, vis1);
					dis[sp_v] = find_mincost_flow(sp_v, t, vis2);
					if (dis[v] + edge[i].cost < dis[sp_v] + edge[j].cost) {
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
				dis[v] = find_mincost_flow(v, t, vis);
				mincost += dis[v] + edge[i].cost;
				vis[v] = true;
			}
		}
	}
	return mincost;
}

static PyObject* mincost_maxflow(PyObject *self, PyObject *args) {
	PyObject *list, *ans;
	Py_ssize_t pos;
	int t, u, v, cap, cost;

	t = (int)PyInt_AsLong(PyTuple_GetItem(args, 0));

	// init edges
	memset(head, -1, sizeof(head));
	e = 0;

	list = PyTuple_GetItem(args, 1);
	for (pos = 0; pos < PyList_Size(list); pos++) {
		PyArg_ParseTuple(PyList_GetItem(list, pos), "iiii", &u, &v, &cap, &cost);
		add_edge(u, v, cap, cost);
		//cout << "add edge:(" << u << " " << v << " " << cap << " " << cost << ")" << endl;
	}

	// init all splited nodes
	split_nodes.clear();
	list = PyTuple_GetItem(args, 2);
	for (pos = 0; pos < PyList_Size(list); pos++) {
		PyArg_ParseTuple(PyList_GetItem(list, pos), "ii", &u, &v);
		split_nodes[u] = v;
		split_nodes[v] = u;
		//cout << "add split nodes:(" << u << " " << v << ")" << endl;
	}

	memset(vis, false, sizeof(vis));
	mincost = find_mincost_flow(0, t, vis);

	ans = PySet_New(NULL);
	for (u = 0; u < t; u++)
		if (vis[u]) PySet_Add(ans, Py_BuildValue("i", u));

	cout << "mincost:" << mincost << endl;
	return ans;
}

static PyMethodDef MaxFlowMethods[] = {
	{"mincost_maxflow", mincost_maxflow, METH_VARARGS, NULL},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initmaxflow(void) {
	PyObject *m;
	m = Py_InitModule("maxflow", MaxFlowMethods);
	if (m == NULL) return;
}
