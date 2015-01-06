#include <Python.h>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <list>

const int eMax = 1000000;
const int nMax = 5000;
const int INF = 1000000000;

struct Edge {
    int u, v, next, prev;
    long cost;
} edge[eMax];

std::unordered_map<int, int> split_nodes;
int e, t, head[nMax], tail[nMax];
long mincost, dis[nMax];
bool vis[nMax];

void add_edge(int u, int v, long cost) {
	edge[e].u = u; edge[e].v = v; edge[e].cost = cost;
    edge[e].next = head[u]; head[u] = e;
    edge[e].prev = tail[v]; tail[v] = e;
    e++;
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
	memset(tail, -1, sizeof(tail));
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

	printf("mincost:%ld\n", mincost);
	return ans;
}

bool visit_graph(int s, int t, bool* vis) {
	if (s == t) return true;
    if (vis[s]) return true;

	bool visited = false;
	for (int i = head[s]; i != -1; i = edge[i].next) {
		if (visit_graph(edge[i].v, t, vis)) {
			//printf("visit edge:%d %d %d\n", s, edge[i].v, edge[i].cost);
			vis[edge[i].v] = true;
			mincost += edge[i].cost;
			visited = true;
		}
	}
	return visited;
}

void remove_edge(int e) {
	//printf("remove edge %d: %d %d\n", e, edge[e].u, edge[e].v);
	if (head[edge[e].u] == e) head[edge[e].u] = edge[e].next;
	else {
		for (int ei = head[edge[e].u]; ei != -1; ei = edge[ei].next) {
			if (edge[ei].next == e) {
				edge[ei].next = edge[e].next;
				break;
			}
		}
	}
	if (tail[edge[e].v] == e) tail[edge[e].v] = edge[e].prev;
	else {
		for (int ei = tail[edge[e].v]; ei != -1; ei = edge[ei].prev) {
			if (edge[ei].prev == e) {
				edge[ei].prev = edge[e].prev;
			}
		}
	}
}

void remove_node(int u) {
	if (!vis[u]) return;
	//printf("remove node: %d\n", u);
	vis[u] = false;

	int sp_u = (split_nodes.find(u) != split_nodes.end()) ? split_nodes[u] : t+1;
    for (int e = head[u]; e != -1; e = edge[e].next) {
    	bool found = false;
		for (int i = head[sp_u]; i != -1; i = edge[i].next) {
			if (edge[i].v == edge[e].v) {
				found = true;
				break;
			}
		}
		remove_edge(e);
    	if (!found) remove_node(edge[e].v);
    }

    for (int e = tail[u]; e != -1; e = edge[e].prev) {
    	bool found = false;
    	for (int i = head[edge[e].u]; i != -1; i = edge[i].next) {
    		if (edge[i].v == sp_u) {
    			found = true;
    			break;
    		}
    	}
    	remove_edge(e);
    	if (!found) remove_node(edge[e].u);
    }
}

static PyObject* maxedge_tiling(PyObject *self, PyObject *args) {
	PyObject *list, *ans;
	Py_ssize_t pos;
	int u, v;
    long cost;

    struct Compare {
    	bool operator()(const std::pair<int, long>& e1, const std::pair<int, long>& e2) const {
    		return e1.second <= e2.second;
    	}
    };
    std::priority_queue<std::pair<int, long>, std::vector<std::pair<int, long>>, Compare> max_heap;

	// init t node
	t = (int)PyInt_AsLong(PyTuple_GetItem(args, 0));

	// init edges
	e = 0;
	memset(head, -1, sizeof(head));
	memset(tail, -1, sizeof(tail));
	list = PyTuple_GetItem(args, 1);
	for (pos = 0; pos < PyList_Size(list); pos++) {
		PyArg_ParseTuple(PyList_GetItem(list, pos), "iik", &u, &v, &cost);
		max_heap.push(std::make_pair(e, cost));
		//printf("add edge %d:(%d, %d, cost=%ld)\n", e, u, v, cost);
		add_edge(u, v, cost);
	}
	max_heap.push(std::make_pair(e, -1));

	// init splited nodes
	split_nodes.clear();
	list = PyTuple_GetItem(args, 2);
	for (pos = 0; pos < PyList_Size(list); pos++) {
		PyArg_ParseTuple(PyList_GetItem(list, pos), "ii", &u, &v);
		split_nodes[u] = v;
		split_nodes[v] = u;
		//printf("add split nodes:(%d, %d)\n", u, v);
	}

	memset(vis, true, sizeof(vis));

	int sp_v, sp_u;
	bool found1, found2, found3;
	std::pair<int, long> ee;
	while ((ee = max_heap.top()).second >= 0) {
		max_heap.pop();

		u = edge[ee.first].u;
		v = edge[ee.first].v;
		sp_v = (split_nodes.find(v) != split_nodes.end()) ? split_nodes[v] : t+1;
		sp_u = (split_nodes.find(u) != split_nodes.end()) ? split_nodes[u] : t+1;

        //printf("max edge %d:%d %d %ld\n", ee.first, u, v, edge[ee.first].cost);

		found1 = found2 = found3 = false;

		for (e = head[u]; e != -1; e = edge[e].next) {
			if (edge[e].v == sp_v) {
				found1 = true;
				break;
			}
		}

		for (e = head[sp_u]; e != -1; e = edge[e].next) {
			if (edge[e].v == v) {
				found2 = true;
				break;
			}
		}

		for (e = head[sp_u]; e != -1; e = edge[e].next) {
			if (edge[e].v == sp_v) {
				found3 = true;
				break;
			}
		}

		if (found1 && vis[sp_v]) {
            //printf("case1: u->sp_v\n");
            remove_edge(ee.first);
			if (!found2) remove_node(v);
			continue;
		}

		if (found2 && vis[sp_u]) {
			//printf("case2: sp_u->v\n");
			remove_edge(ee.first);
			remove_node(u);
			continue;
		}

		if (found3 && vis[sp_u] && vis[sp_v]) {
			//printf("case3: sp_u->sp_v\n");
			remove_edge(ee.first);
			remove_node(u);
			remove_node(v);
			continue;
		}
        //printf("do no remove edge:%d %d\n", u, v);
	}

	memset(vis, false, sizeof(vis));
	mincost = 0;
	visit_graph(0, t, vis);

	ans = PyList_New(0);
	for (u = 0; u < t; u++)
		if (vis[u]) PyList_Append(ans, Py_BuildValue("i", u));

	printf("maxedge:%ld\n", mincost);
	return ans;
}

bool choose_nodes[nMax], visited[nMax], isBest;

long calc_cost(int s, int t) {
	if (s == t || visited[s]) return 0;

	long cost = 0;
	for (int i = head[s]; i != -1; i = edge[i].next) {
		if (choose_nodes[edge[i].v]) {
			cost += calc_cost(edge[i].v, t) + edge[i].cost;
		}
	}
	visited[s] = true;
	return cost;
}

void find_solution(int size) {
	if (size == 0) {
		memset(visited, false, sizeof(visited));
		long cost = calc_cost(0, t);
		if (!(isBest ^ (cost < mincost))) {
			mincost = cost;
			memcpy(vis, choose_nodes, t * sizeof(bool));
		}
	} else {
        std::unordered_map<int, int>::iterator it = split_nodes.begin();
		int u = (*it).first, v = (*it).second;

		split_nodes.erase(u);
		split_nodes.erase(v);

		choose_nodes[u] = false;
		find_solution(size - 2);

		choose_nodes[u] = true;
		choose_nodes[v] = false;
		find_solution(size - 2);

		choose_nodes[v] = true;
		split_nodes[u] = v;
		split_nodes[v] = u;
	}
}

static PyObject* best_tiling(PyObject *self, PyObject *args) {
	PyObject *list, *ans;
	Py_ssize_t pos;
	int u, v;
    long cost;

	// init t node
	t = (int)PyInt_AsLong(PyTuple_GetItem(args, 0));

	// init edges
	e = 0;
	memset(head, -1, sizeof(head));
	memset(tail, -1, sizeof(tail));
	list = PyTuple_GetItem(args, 1);
	for (pos = 0; pos < PyList_Size(list); pos++) {
		PyArg_ParseTuple(PyList_GetItem(list, pos), "iik", &u, &v, &cost);
		//printf("add edge %d:(%d, %d, cost=%ld)\n", e, u, v, cost);
		add_edge(u, v, cost);
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
	mincost = 2147483647L;
	isBest = true;

	memset(choose_nodes, true, sizeof(choose_nodes));
	find_solution(split_nodes.size());

	ans = PyList_New(0);
	for (u = 1; u < t; u++)
		if (vis[u]) PyList_Append(ans, Py_BuildValue("i", u));

	printf("best:%ld\n", mincost);
	return ans;
}

static PyObject* worse_tiling(PyObject *self, PyObject *args) {
	PyObject *list, *ans;
	Py_ssize_t pos;
	int u, v;
    long cost;

	// init t node
	t = (int)PyInt_AsLong(PyTuple_GetItem(args, 0));

	// init edges
	e = 0;
	memset(head, -1, sizeof(head));
	memset(tail, -1, sizeof(tail));
	list = PyTuple_GetItem(args, 1);
	for (pos = 0; pos < PyList_Size(list); pos++) {
		PyArg_ParseTuple(PyList_GetItem(list, pos), "iik", &u, &v, &cost);
		//printf("add edge %d:(%d, %d, cost=%ld)\n", e, u, v, cost);
		add_edge(u, v, cost);
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
	mincost = -1;
	isBest = false;

	memset(choose_nodes, true, sizeof(choose_nodes));
	find_solution(split_nodes.size());

	ans = PyList_New(0);
	for (u = 1; u < t; u++)
		if (vis[u]) PyList_Append(ans, Py_BuildValue("i", u));

	printf("worse:%ld\n", mincost);
	return ans;
}

static PyMethodDef TilingMethods[] = {
	{"mincost_tiling", mincost_tiling, METH_VARARGS, NULL},
	{"maxedge_tiling", maxedge_tiling, METH_VARARGS, NULL},
	{"best_tiling", best_tiling, METH_VARARGS, NULL},
	{"worse_tiling", worse_tiling, METH_VARARGS, NULL},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC inittiling(void) {
	PyObject *m;
	m = Py_InitModule("tiling", TilingMethods);
	if (m == NULL) return;
}

