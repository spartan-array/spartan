#include <Python.h>
#include <unordered_map>
#include <map>
#include <queue>
#include <algorithm>
#include <list>
#include <set>
#include <iostream>

#define NUM_NODE_PER_GROUP 4
const int eMax = 1000000;
const int nMax = 5000;
const int INF = 1000000000;

struct Edge {
    int u, v, next, prev;
    long cost;
} edge[eMax];

int e, t, head[nMax], tail[nMax];
long mincost, dis[nMax];
bool vis[nMax];
bool visited_groups[nMax];
bool valid_nodes[nMax];

int group_id, groups[nMax][NUM_NODE_PER_GROUP];
std::unordered_map<int, int> split_nodes;

void add_edge(int u, int v, long cost) {
    edge[e].u = u; edge[e].v = v; edge[e].cost = cost;
    edge[e].next = head[u]; head[u] = e;
    edge[e].prev = tail[v]; tail[v] = e;
    e++;
}

int compare(const void *e1, const void *e2) {
	return *((int *)(e1)) - *((int *)(e2));
}

void init_graph(PyObject *args) {
	PyObject *list;
	Py_ssize_t pos;
	int v[NUM_NODE_PER_GROUP];
    long cost;

	// init t node
	t = (int)PyInt_AsLong(PyTuple_GetItem(args, 0));
    printf("number of nodes:%d\n", t);

	// init edges
	e = 0;
	memset(head, -1, sizeof(head));
	memset(tail, -1, sizeof(tail));
	list = PyTuple_GetItem(args, 1);
	for (pos = 0; pos < PyList_Size(list); pos++) {
		PyArg_ParseTuple(PyList_GetItem(list, pos), "iik", &v[0], &v[1], &cost);
		add_edge(v[0], v[1], cost);
		//printf("add edge:(%d, %d, cost=%ld)\n", v[0], v[1], cost);
	}

	// init splited nodes
	split_nodes.clear();
	group_id = 0;
	list = PyTuple_GetItem(args, 2);
	for (pos = 0; pos < PyList_Size(list); pos++) {
		PyArg_ParseTuple(PyList_GetItem(list, pos), "iiii", &v[0], &v[1], &v[2], &v[3]);
		for (int i = 0; i < NUM_NODE_PER_GROUP; i++) {
			groups[group_id][i] = v[i];
			split_nodes[v[i]] = group_id;
			valid_nodes[v[i]] = true;
		}
		qsort(groups[group_id], NUM_NODE_PER_GROUP, sizeof(int), compare);
		group_id++;
		//printf("add split nodes:(%d, %d, %d, %d, %d, %d)\n", v[0], v[1], v[2], v[3], v[4], v[5]);
	}
}

void print_choices(int s, int t, bool* vis, long mincost) {
	printf("%d choose (cost=%ld):", s, mincost);
	for (int u = 0; u <= t; u++)
		if (vis[u]) printf("%d ", u);
	printf("\n");
}

long find_mincost_tiling(int s, int t, bool* vis) {
	long mincost = 0;
	std::list<int> child_edges;
    int i, j, k, v, sp_v, size, split_edges[NUM_NODE_PER_GROUP];

	for (i = head[s]; i != -1; i = edge[i].next) child_edges.push_back(i);
	size = child_edges.size();

	while (!child_edges.empty()) {
		i = child_edges.front(); child_edges.pop_front(); size --;
		v = edge[i].v;

		if (split_nodes.find(v) != split_nodes.end()) { // splited node
			int vis_v = 0;
			for (k = 0; k < NUM_NODE_PER_GROUP; k++) {
				sp_v = groups[split_nodes[v]][k];

				// find split edge j
				for (j = i; j != -1 && edge[j].v != sp_v; j=edge[j].next);
				split_edges[k] = j;
				if (j >= 0 && sp_v != v) { child_edges.remove(j); size --; }

				if (vis[sp_v]) {
					mincost += (j < 0)? INF: edge[j].cost;
					vis_v ++;
				}
			}

			if (vis_v == 0) {
				bool vis1[nMax], min_vis[nMax];
				int min_v = 0, min_count = 0, min_cost = -1;
				for (k = 0; k < NUM_NODE_PER_GROUP; k++) {
					sp_v = groups[split_nodes[v]][k];
					j = split_edges[k];
					if (j < 0) continue;

					memcpy(vis1, vis, t * sizeof(bool));
					dis[sp_v] = find_mincost_tiling(sp_v, t, vis1);
					if (min_cost < 0 || dis[sp_v] + edge[j].cost < min_cost) {
						min_v = sp_v;
						min_cost = dis[sp_v] + edge[j].cost;
						memcpy(min_vis, vis1, t * sizeof(bool));
						min_count = 1;
					} else if (dis[sp_v] + edge[j].cost == min_cost) {
						min_count ++;
					}
				}

				if (min_count == 1 || size <= 0) {
					mincost += min_cost;
					memcpy(vis, min_vis, t * sizeof(bool));
					vis[min_v] = true;
				} else {
					for (k = 0; k < NUM_NODE_PER_GROUP; k++) {
						if (split_edges[k] >= 0) child_edges.push_back(split_edges[k]);
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
    init_graph(args);

	memset(vis, false, sizeof(vis));
	mincost = find_mincost_tiling(0, t, vis);

	PyObject *ans = PyList_New(0);
	for (int u = 1; u < t; u++)
		if (vis[u]) PyList_Append(ans, Py_BuildValue("i", u));

	printf("mincost allcost:%ld\n", mincost);
	return ans;
}

bool choose_nodes[nMax], visited[nMax], isBest;

long calc_cost(int s, int t) {
	if (s == t || visited[s]) return 0;

	long cost = 0, v_cost;
	for (int i = head[s]; i != -1; i = edge[i].next) {
		if (choose_nodes[edge[i].v]) {
            v_cost = calc_cost(edge[i].v, t);
            if (v_cost < 0) {
                cost = v_cost;
                break;
            }
			cost += v_cost + edge[i].cost;
		} else if (split_nodes.find(edge[i].v) != split_nodes.end()) {
			int k, choose_v;
			for (k = 0; k < NUM_NODE_PER_GROUP; k++) {
				choose_v = groups[split_nodes[edge[i].v]][k];
				if (choose_nodes[choose_v]) break;
			}
			for (k = head[s]; k != -1 && edge[k].v != choose_v; k = edge[k].next);
			if (k == -1) {
				cost = -INF;
				break;
			}
		} else {
			cost = -INF;
			break;
		}
	}

	visited[s] = true;
	return cost;
}

long view_cost(int u, int j, bool is_max_cost=false) {
	if (split_nodes.find(edge[j].v) == split_nodes.end()) return edge[j].cost;

	int i, edge_count = 0;
	long r_cost = edge[j].cost;
	for (i = head[u]; i != -1; i = edge[i].next)
		if (split_nodes.find(edge[i].v) != split_nodes.end() && split_nodes[edge[i].v] == split_nodes[edge[j].v]) {
			edge_count ++;
			if (is_max_cost ^ (edge[i].cost < r_cost)) r_cost = edge[i].cost;
		}

	if (edge_count == 1 && edge[j].cost == 0) {
        //printf("%d %d is view node\n", u, edge[j].v);
		r_cost = 0;
		std::set<int> visited_groups;
		for (i = head[edge[j].v]; i != -1; i = edge[i].next) {
			if (split_nodes.find(edge[i].v) == split_nodes.end())
				r_cost += view_cost(edge[j].v, i);
			else if (visited_groups.find(split_nodes[edge[i].v]) == visited_groups.end()) {
				r_cost += view_cost(edge[j].v, i);
				visited_groups.insert(split_nodes[edge[i].v]);
			}
		}
        return r_cost;
	}
	return r_cost;
}

void remove_edge(int i) {
	int u = edge[i].u, v = edge[i].v, j;

	if (head[u] == i) head[u] = edge[i].next;
    else {
	    for (j = head[u]; edge[j].next != -1 && edge[j].next != i; j = edge[j].next);
        edge[j].next = edge[i].next;
    }
	if (tail[v] == i) tail[v] = edge[i].prev;
    else {
	    for (j = tail[v]; edge[j].prev != -1 && edge[j].prev != i; j = edge[j].prev);
	    edge[j].prev = edge[i].prev;
    }
}

void remove_node(int node) {
	vis[node] = false;

	int i;
	for (i = head[node]; i != -1; i = edge[i].next) {
		remove_edge(i);
		if (tail[edge[i].v] == -1) remove_node(edge[i].v);
	}

	for (i = tail[node]; i != -1; i = edge[i].prev) {
		remove_edge(i);
		if (head[edge[i].u] == -1) remove_node(edge[i].u);
	}
}

static PyObject* maxedge_tiling(PyObject *self, PyObject *args) {
	init_graph(args);

    struct Compare {
    	bool operator()(const std::pair<int, std::pair<int, long>>& e1,
    			        const std::pair<int, std::pair<int, long>>& e2) const {
    		if (e1.second.first != e2.second.first)
    			return e1.second.first <= e2.second.first;

    		return e1.second.second <= e2.second.second;
    	}
    };
    std::priority_queue<std::pair<int, std::pair<int, long>>,
						std::vector<std::pair<int, std::pair<int, long>>>, Compare> max_heap;

    int i, j, k, u, g;
    long cost;

    std::set<int> visited_groups;
    for (i = 0; i < group_id; i++) {
    	cost = 0;
    	for (k = 0; k < NUM_NODE_PER_GROUP; k++) {
    		u = groups[i][k];
    		for (j = head[u]; j != -1; j = edge[j].next) {
    			g = (split_nodes.find(edge[j].v) != split_nodes.end())? split_nodes[edge[j].v] : edge[j].v + group_id;
    			if (visited_groups.find(g) == visited_groups.end()) {
    				cost += view_cost(u, j, true);
    				visited_groups.insert(g);
    			}
    		}
    		for (j = tail[u]; j != -1; j = edge[j].prev) {
    			g = (split_nodes.find(edge[j].u) != split_nodes.end())? split_nodes[edge[j].u] : edge[j].u + group_id;
				cost += edge[j].cost;
				visited_groups.insert(g);
    		}
    	}
    	max_heap.push(std::make_pair(i, std::make_pair(visited_groups.size(), cost)));
    	visited_groups.clear();
    }
    max_heap.push(std::make_pair(group_id, std::make_pair(0, -1)));

	memset(vis, true, sizeof(vis));

	int min_u;
	long min_cost;
	std::pair<int, std::pair<int, long>> u_cost;
	while ((u_cost = max_heap.top()).second.second >= 0) {
		max_heap.pop();

        min_u = 0;
		min_cost = -1;
		for (k = 0; k < NUM_NODE_PER_GROUP; k++) {
			u = groups[u_cost.first][k];
			if (vis[u] == false) continue;

			vis[u] = false;
			cost = 0;
			visited_groups.clear();
			for (j = head[u]; j != -1; j = edge[j].next) {
				g = (split_nodes.find(edge[j].v) != split_nodes.end())? split_nodes[edge[j].v] : edge[j].v + group_id;
				if (visited_groups.find(g) == visited_groups.end()) {
					cost += view_cost(u, j);
					visited_groups.insert(g);
				}
			}

			visited_groups.clear();
			for (j = tail[u]; j != -1; j = edge[j].prev) {
				if (split_nodes.find(edge[j].u) == split_nodes.end())
					cost += edge[j].cost;
				else if (visited_groups.find(split_nodes[edge[j].u]) == visited_groups.end()) {
					long m_cost = edge[j].cost;
					for (i = tail[u]; i != -1; i = edge[i].prev)
						if (split_nodes.find(edge[i].u) != split_nodes.end() &&
							split_nodes[edge[i].u] == split_nodes[edge[j].u] &&
							edge[j].cost < m_cost)
							m_cost = edge[j].cost;
					cost += m_cost;
					visited_groups.insert(split_nodes[edge[j].u]);
				}
			}

			if (min_cost < 0 || cost < min_cost) {
				min_cost = cost;
				min_u = u;
			}
		}
        printf("max group:%d %ld choose u:%d %ld\n", u_cost.first, u_cost.second.second, min_u, min_cost);
		vis[min_u] = true;
		for (k = 0; k < NUM_NODE_PER_GROUP; k++) {
			u = groups[u_cost.first][k];
			if (u != min_u) remove_node(u);
		}
	}

	memcpy(choose_nodes, vis, sizeof(choose_nodes));
	memset(visited, false, sizeof(visited));
    mincost = calc_cost(0, t);

	PyObject *ans = PyList_New(0);
	for (u = 1; u < t; u++)
		if (vis[u]) PyList_Append(ans, Py_BuildValue("i", u));

	printf("maxedge allcost:%ld\n", mincost);
	return ans;
}


void find_solution(int id) {
	if (id == group_id) {
		memset(visited, false, sizeof(visited));
		long cost = calc_cost(0, t);
		if (cost >= 0 && !(isBest ^ (cost < mincost))) {
			mincost = cost;
			memcpy(vis, choose_nodes, t * sizeof(bool));
		}
	} else {
        for (int k = 0; k < NUM_NODE_PER_GROUP; k++) {
        	choose_nodes[groups[id][k]] = false;
        }

        for (int k = 0; k < NUM_NODE_PER_GROUP; k++) {
        	choose_nodes[groups[id][k]] = true;
        	find_solution(id+1);
        	choose_nodes[groups[id][k]] = false;
        }
	}
}

static PyObject* best_tiling(PyObject *self, PyObject *args) {
	init_graph(args);

	memset(vis, false, sizeof(vis));
	mincost = 2147483647L;
	isBest = true;

	memset(choose_nodes, true, sizeof(choose_nodes));
	find_solution(0);

	PyObject *ans = PyList_New(0);
	for (int u = 1; u < t; u++)
		if (vis[u]) PyList_Append(ans, Py_BuildValue("i", u));

	printf("best allcost:%ld\n", mincost);
	return ans;
}

static PyObject* worse_tiling(PyObject *self, PyObject *args) {
	init_graph(args);

	memset(vis, false, sizeof(vis));
	mincost = -1;
    isBest = false;

	memset(choose_nodes, true, sizeof(choose_nodes));
	find_solution(0);

	PyObject *ans = PyList_New(0);
	for (int u = 1; u < t; u++)
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

