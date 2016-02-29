#ifndef __DBSCAN_NONSPARSE_H__
#define __DBSCAN_NONSPARSE_H__

#include "dbscan.h"

namespace libdbscan {

template <typename TNum>
class dbscan_nonsparse : public dbscan<TNum> {
public:
    // Corpus is expected to be a C-style (row-major) 2 dimensional array. 
    //
    // We use unsafe buffers here for max interop with other libraries (e.g.
    // ease of being able to take a view of some other lib's buffer without
    // doing any copies) - perhaps could be improved with std::array
    dbscan_nonsparse(const TNum* corpus, index_t rows, index_t cols);
    ~dbscan_nonsparse() {}
protected:
    typedef const TNum* corpus_vector_t;
    virtual index_t region_query(index_t vec_i, TNum ps, index_set& result) override;
    corpus_vector_t _corpus;
};

template <typename TNum>
dbscan_nonsparse<TNum>::dbscan_nonsparse(const TNum* corpus, index_t rows, index_t cols) : 
    _corpus(corpus)
{
    // init protected members
    dbscan<TNum>::_rows = rows;
    dbscan<TNum>::_cols = cols;
}

template <typename TNum>
index_t dbscan_nonsparse<TNum>::region_query(index_t vec_i, TNum eps, index_set& result) 
{
    // return a list of indexes of corpus vectors that are < eps away from
    // vec_i, according to euclidian distance, sorted ascending
    eps = eps * eps;
    
    const TNum* comparison_vector = &_corpus[vec_i * this->_cols];
    for (index_t i=0; i < this->_rows; i++) {
        if (i == vec_i) {
            continue;
        }

        const TNum* row = &_corpus[i*this->_cols];
        TNum distance = euclidean_distance<TNum>(this->_cols, row, comparison_vector);
        if (distance <= eps) {
            result.insert(i);
        }
    }    

    return result.size();
}

}

#endif
