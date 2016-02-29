#ifndef __DBSCAN_H__
#define __DBSCAN_H__

#include <cmath>
#include <cstring>
#include <ctime>
#include <memory>
#include <stdio.h>
#include <unordered_set>
#include <vector>

#include "util.h"

namespace libdbscan {

// The type of indexes used throughout; seems sensible to make it 64-bit on
// 64-bit platforms; on LP64 this works, would need extra love for windows.
typedef long index_t;

// Expected to be a hash-based set with O(1) operations
typedef std::unordered_set<index_t> index_set;

template <typename TNum>
class dbscan {
    // Abstract base class for dbscan implementations.
    //
    // This assumes a model in which the constructor is passed a corpus of vector
    // data and holds it as internal data; the caller then calls run() to
    // execute the algorithm on the results.
    //
    // Concrete subclasses should implement 
    //
    //  * Their own constructor, which must initialize at _rows and _cols
    //  * region_query.
public:
    void run(TNum eps, index_t min_pts, std::vector<index_t>& results, 
        std::vector<index_t>& noise);
    index_t get_num_rows() { return _rows; }
    virtual ~dbscan() {}

protected:
    // Subclasses must implement this. Given the index of a vector in the
    // corpus, fill result with a set of indexes of vectors that are within eps
    // distance, according to the distance metric being used. Returns the size
    // of the result.
    virtual index_t region_query(index_t vec_i, TNum eps, index_set& result) = 0;
    index_t _rows;
    index_t _cols;

private:
    void expand_cluster(TNum eps, 
        index_t min_pts,
        index_t cluster_i, 
        index_t vec_i, 
        index_set& neighbour_pts,
        index_set& visited,
        std::vector<index_t>& results);

    void expand_cluster_inner(TNum eps,
        index_t min_pts,
        index_t cluster_i, 
        index_t vec_i, 
        index_set& pts,
        index_set& visited,
        std::vector<index_t>& results,
        index_set& additional_pts);
};

template <typename TNum>
void dbscan<TNum>::run(TNum eps, index_t min_pts, std::vector<index_t>& results, 
        std::vector<index_t>& noise)
{
    // Fairly literal implementation of the outer function of DBSCAN
    results.resize(_rows, -1);
    noise.resize(_rows, 0);
    index_set visited;

    index_t cluster_i = -1;

    for (index_t i=0; i < _rows; i++) {
        if (visited.find(i) != visited.end()) {
            continue;
        }

        visited.insert(i);

        index_set query_result;
        int num_in_region = region_query(i, eps, query_result);

        if (num_in_region < min_pts) {
            noise[i] = true;
            continue;
        }

        cluster_i++;
        expand_cluster(eps, min_pts, cluster_i, i, query_result, visited, results);
    }
}

template <typename TNum>
void dbscan<TNum>::expand_cluster(TNum eps, 
    index_t min_pts,
    index_t cluster_i, 
    index_t vec_i, 
    index_set& neighbour_pts,
    index_set& visited,
    std::vector<index_t>& results)
{
    results[vec_i] = cluster_i;

    auto additional_pts = std::make_unique<index_set>();
    expand_cluster_inner(eps, min_pts, cluster_i, vec_i, neighbour_pts, 
            visited, results, *additional_pts);

    while (additional_pts->size()) {
        auto next_additional_pts = std::make_unique<index_set>();
        expand_cluster_inner(eps, min_pts, cluster_i, vec_i, *additional_pts, 
                visited, results, *next_additional_pts);
        additional_pts.swap(next_additional_pts);
    }
}

template <typename TNum>
void dbscan<TNum>::expand_cluster_inner(TNum eps,
    index_t min_pts,
    index_t cluster_i, 
    index_t vec_i, 
    index_set& pts,
    index_set& visited,
    std::vector<index_t>& results,
    index_set& additional_pts)
{
    // Fairly literal implementation of the inner function of DBSCAN
    for (const auto& pt_i : pts) {
        if (visited.find(pt_i) == visited.end()) {
            visited.insert(pt_i);

            index_set region_query_results;
            int num_close_neighbours = region_query(
                pt_i, 
                eps, 
                region_query_results);

            if (num_close_neighbours >= min_pts) {
                for (const auto& rq_i : region_query_results) {
                    // The algorithm calls for the pts to be merged with the
                    // set we're currently iterating over, which isn't possible
                    // with c++ containers - so we accumulate them in
                    // additional pts, which if non-empty will cause the caller
                    // to call this function again, passing it in.
                    // 
                    // You could also just recurse, but this way saves stack
                    // space.
                    if (pts.find(rq_i) == pts.end()) {
                        additional_pts.insert(rq_i);
                    }
                }
            }
        }

        if (results[pt_i] == -1) {
            results[pt_i] = cluster_i;
        }
    }
};

}
#endif
