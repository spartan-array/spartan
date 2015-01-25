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

	printf("mincost:%ld\n", mincost);
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

std::map<std::vector<int>, bool > view_pairs;
void init_view_pairs(void)
{
    int i, j, u, k;
    for (i = 0; i < group_id; i++) {
        for (j = i + 1; j < group_id; j++) {
            bool has_matched[NUM_NODE_PER_GROUP];
            bool is_view = true;
            for (k = 0 ; k < NUM_NODE_PER_GROUP; k++) 
                has_matched[k] = false;
            for (k = 0 ; k < NUM_NODE_PER_GROUP && is_view; k++) {
    		    u = groups[i][k];
                for (int e = head[u]; e != -1 && is_view; e = edge[e].next) {
                    if (split_nodes.find(edge[e].v) != split_nodes.end() &&
                        split_nodes[edge[e].v] == j) {
                        int l;
                        for (l = 0 ; l < NUM_NODE_PER_GROUP ; l++) 
                            if (groups[j][l] == edge[e].v)
                                break;
                        if (not has_matched[l]) {
                            has_matched[l] = true;
                        } else {
                            is_view = false;
                        }
                        break;
                    }
                }
            }
            for (k = 0 ; k < NUM_NODE_PER_GROUP; k++) 
                if (not has_matched[k])
                    is_view = false;
            if (is_view) {
                std::vector<int> pair;
                pair.push_back(i); 
                pair.push_back(j);
                //printf("view node %d %d\n", i, j);
                view_pairs.insert(std::make_pair(pair, true));
            }
        }
    }
}

long view_cost(int u, int j) {
    /*
    std::vector<int> pair;
    pair.push_back(split_nodes[u]);
    pair.push_back(split_nodes[edge[j].v]);
    if (pair[0] > pair[1]) {
        int t = pair[1];
        pair[1] = pair[0];
        pair[0] = t;
    }
    if (view_pairs.find(pair) == view_pairs.end()) {
        return 0;
    }
    */

    if (split_nodes.find(edge[j].v) == split_nodes.end()) return edge[j].cost;

    int i, edge_count = 0;
    for (i = head[u]; i != -1; i = edge[i].next) {
        if (split_nodes.find(edge[i].v) != split_nodes.end() && split_nodes[edge[i].v] == split_nodes[edge[j].v])
            edge_count ++;
    }

	if (edge_count == 1 && edge[j].cost == 0) {
        //printf("%d %d is view node\n", split_nodes[u], split_nodes[edge[j].v]);
		long cost = 0;
		for (int l = head[edge[j].v]; l != -1; l = edge[l].next)
			cost += view_cost(edge[j].v, l);
        return cost;
	}
	return edge[j].cost;
	//return 0;
}

static PyObject* maxedge_tiling(PyObject *self, PyObject *args) {
	init_graph(args);
    init_view_pairs();

    struct Compare {
    	bool operator()(const std::pair<int, std::vector<long> >& e1, const std::pair<int, std::vector<long> >& e2) const {
    	    if (e1.second[0] < e2.second[0])
    	        return true;
            else if (e1.second[0] > e2.second[0])
                return false;
            else 
                return e1.second[1] <= e2.second[1];
    	}
    };
    std::priority_queue<std::pair<int, std::vector<long> >, std::vector<std::pair<int, std::vector<long> > >, Compare> max_heap;

    int i, j, k, u;
    long cost;

    std::vector<std::map<int, int> > connectivity_table;
    std::vector<int> connectivity;
    for (i = 0; i < group_id; i++) {
        connectivity.push_back(0);
        std::map <int, int> m;
        connectivity_table.push_back(m);
    }
        
    for (i = 0; i < e ; i++) {
        int u = edge[i].u;
        int v = edge[i].v;
        // Ignore source and terminal
        if (u == 0 or v == 0 or head[u] == -1 or head[v] == -1)
            continue;
        while (split_nodes.find(u) == split_nodes.end()) {
            u = edge[head[u]].v;
        }
        while (split_nodes.find(v) == split_nodes.end()) {
            v = edge[head[v]].v;
        }
        u = split_nodes[u];
        v = split_nodes[v];
        if (u < v) {
            connectivity_table[u][v] = 1;
        } else if (u > v) {
            connectivity_table[v][u] = 1;
        }
    }
    for (i = 0; i < group_id; i++) {
        connectivity[i] += connectivity_table[i].size();
        for (auto it = connectivity_table[i].begin(); it != connectivity_table[i].end(); ++it) 
            connectivity[it->first] += 1;
    }

    for (i = 0; i < group_id; i++) {
    	cost = 0;
    	for (k = 0; k < NUM_NODE_PER_GROUP; k++) {
    		u = groups[i][k];
            for (j = head[u]; j != -1; j = edge[j].next) cost += view_cost(u, j);
    		for (j = tail[u]; j != -1; j = edge[j].prev) cost += edge[j].cost;
    	}
        std::vector<long> weight;
        weight.push_back(connectivity[i]);
        printf("coonectivity %d\n", connectivity[i]);
        weight.push_back(cost);
    	max_heap.push(std::make_pair(i, weight));

    }
    std::vector<long> weight;
    weight.push_back(0);
    weight.push_back(-1);
    max_heap.push(std::make_pair(group_id, weight));

	memset(vis, true, sizeof(vis));
	memset(visited_groups, false, sizeof(visited_groups));

	int min_u;
	long min_cost;
	std::pair<int, std::vector<long> > u_cost;
	while ((u_cost = max_heap.top()).second[1] >= 0) {
		max_heap.pop();

        min_u = -1;
		min_cost = -1;
		for (k = 0; k < NUM_NODE_PER_GROUP; k++) {
			u = groups[u_cost.first][k];
			vis[u] = false;

			cost = 0;
            /*
			for (j = head[u]; j != -1; j = edge[j].next) {
				if (!vis[edge[j].v]) {
					for (i = 0; i < NUM_NODE_PER_GROUP && !vis[groups[split_nodes[edge[j].v]][i]]; i++);
					int choose_v = groups[split_nodes[edge[j].v]][i];
					for (i = head[u]; i != -1 && edge[i].v != choose_v; i = edge[i].next);

					if (i == -1) break; else continue;
				}
				cost += view_cost(u, j);
			}

			if (j != -1) continue;

			for (j = tail[u]; j != -1; j = edge[j].prev) {
				if (!vis[edge[j].u]) {
					for (i = 0; i < NUM_NODE_PER_GROUP && !vis[groups[split_nodes[edge[j].u]][i]]; i++);
					int choose_u = groups[split_nodes[edge[j].u]][i];
					for (i = tail[u]; i != -1 && edge[i].u != choose_u; i = edge[i].prev);

					if (i == -1) break; else continue;
				}
				if (u_cost.first == 5 and k == 3) {
				    printf("==> %d %ld\n", j, edge[j].cost);
                }
				cost += edge[j].cost;
			}
			*/

            std::unordered_map<int, bool> caculated_groups;
            int hedge, tedge;
            for (hedge = head[u]; hedge != -1; hedge = edge[hedge].prev) {
                int to_group = split_nodes[edge[hedge].v];
                if (caculated_groups.find(to_group) != caculated_groups.end())
                    continue;
                if (visited_groups[to_group]) {
                    int candidate;
                    for (candidate = head[u]; candidate != -1 ; candidate = edge[candidate].prev) {
                        if (split_nodes[edge[candidate].v] == to_group &&
                                        vis[edge[candidate].v]) {
                            break;
                        }
                    }
                    if (candidate == -1) {
                        cost = -1;
                        break;
                    }
                    cost += edge[candidate].cost;
                    caculated_groups[to_group] = true;
                } else {
                    int candidate_cost = INT_MAX;
                    for (int c = head[u]; c != -1 ; c = edge[c].next) {
                        if (split_nodes[edge[c].v] == to_group &&
                            view_cost(u, c) < candidate_cost) {
                            candidate_cost = edge[c].cost;
                        }
                    }
                    cost += candidate_cost;
                    caculated_groups[to_group] = true;
                }
            }
			if (caculated_groups.size() == 0 or cost == -1) 
			    continue;

            caculated_groups.clear();
            for (tedge = tail[u]; tedge != -1; tedge = edge[tedge].prev) {
                int from_group = split_nodes[edge[tedge].u];
                if (caculated_groups.find(from_group) != caculated_groups.end())
                    continue;
                if (visited_groups[from_group]) {
                    int candidate;
                    for (candidate = tail[u]; candidate != -1 ; candidate = edge[candidate].prev) {
                        if (split_nodes[edge[candidate].u] == from_group &&
                            vis[edge[candidate].u]) {
                            break;
                        }
                    }
                    if (candidate == -1) {
                        cost = -1;
                        break;
                    }
                    cost += edge[candidate].cost;
                    caculated_groups[from_group] = true;
                } else {
                    int candidate_cost = INT_MAX;
                    for (int c = tail[u]; c != -1 ; c = edge[c].prev) {
                        if (split_nodes[edge[c].u] == from_group && 
                            edge[c].cost < candidate_cost) {
                            candidate_cost = edge[c].cost;
                        }
                    }
                    cost += candidate_cost;
                    caculated_groups[from_group] = true;
                }
            }
			if (caculated_groups.size() == 0 or cost == -1) 
			    continue;

            //printf("max group %d tilng %d cost = %ld\n", u_cost.first, k, cost);
			if (min_cost < 0 || cost < min_cost) {
				min_cost = cost;
				min_u = u;
			}
		}
		visited_groups[u_cost.first] = true;
        printf("max group:%d choose u:%d %ld\n", u_cost.first,  min_u, min_cost);
		vis[min_u] = true;
	}

	memcpy(choose_nodes, vis, sizeof(choose_nodes));
	memset(visited, false, sizeof(visited));
    mincost = calc_cost(0, t);

	PyObject *ans = PyList_New(0);
	for (u = 1; u < t; u++)
		if (vis[u]) PyList_Append(ans, Py_BuildValue("i", u));

	printf("maxedge:%ld\n", mincost);
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

	printf("best:%ld\n", mincost);
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

