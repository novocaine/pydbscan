#ifndef __DBSCAN_SPARSE_H__
#define __DBSCAN_SPARSE_H__

// Use boost bindings for ublas sparse arrays - not sure this is maintained
// all that well, isn't available in debian for example
#include <boost/numeric/ublas/vector_sparse.hpp>

#include "dbscan.h"

namespace libdbscan {

template <typename T>
using corpus_vector_t = boost::numeric::ublas::compressed_vector<T>;

template <typename TNum>
struct euclidean_distance_metric {
    // Classic Euclidean distance of [ (x_1 - y_1)^2 + .. + (x_n - y_n)^2 ] ^ 0.5
    //
    // Similar distance metrics are also possible by parametrising the outer
    // exponent as in minkowski distance, but this is not yet implemented
    TNum _eps;

    euclidean_distance_metric(TNum eps) {
        _eps = eps * eps;
    }

    bool operator () (const corpus_vector_t<TNum>& x, 
        const corpus_vector_t<TNum>& y)
    {
        TNum sum = 0;

        auto x_iter = x.begin();
        auto y_iter = y.begin();

        if (x.begin() == x.end() || y.begin() == y.end()) {
            // one or the other of the vectors is empty, so can't really
            // compare..
            return false;
        }

        while (true) {
            if (x_iter==x.end()) {
                while (y_iter != y.end()) {
                    TNum n = *y_iter;
                    sum += n*n;
                    ++y_iter;
                }
                return sum <= _eps;
            }
            if (y_iter==y.end()) {
                while (x_iter != x.end()) {
                    TNum n = *x_iter;
                    sum += n*n;
                    ++x_iter;
                }
                return sum <= _eps;
            }

            if (x_iter.index() < y_iter.index()) {
                TNum n = *y_iter;
                sum += n*n;
                ++x_iter;
            } else if (y_iter.index() < x_iter.index()) {
                TNum n = *x_iter;
                sum += n*n;
                ++y_iter;
            } else { 
                TNum d = *x_iter - *y_iter;
                sum += d*d;
                ++x_iter;
                ++y_iter;
            }
        }
    }
};

template <typename TNum>
struct cosine_similarity_metric {
    // Cosine similarity is a measure of the angle between the two vectors; it 
    // measures relative orientation and ignores magnitude.
    //
    // A similarity of 1 indicates the vectors have the same orientation;
    // a similarity of 0 indicates the vectors are orthogonal. So unlike
    // Euclidean or other measures of distance, greater values (up to 1)
    // indicate higher similarity.
    TNum _eps;

    cosine_similarity_metric(TNum eps) {
        _eps = eps;
    }

    bool operator () (const corpus_vector_t<TNum>& x, 
            const corpus_vector_t<TNum>& y) {
        TNum sum = 0;
        TNum sum_x = 0;
        TNum sum_y = 0;

        auto x_iter = x.begin();
        auto y_iter = y.begin();

        while (true) {
            if (x_iter==x.end()) {
                TNum denom = sqrt(sum_x) * sqrt(sum_y);
                return (denom == 0 ? 0 : sum / denom) > _eps;
            }
            if (y_iter==y.end()) {
                TNum denom = sqrt(sum_x) * sqrt(sum_y);
                return (denom == 0 ? 0 : sum / denom) > _eps;
            }

            if (x_iter.index() < y_iter.index()) {
                sum_x += ((*x_iter) * (*x_iter));
                ++x_iter;
            } else if (y_iter.index() < x_iter.index()) {
                sum_y += ((*y_iter) * (*y_iter));
                ++y_iter;
            } else { 
                sum += ((*x_iter) * (*y_iter));
                sum_x += ((*x_iter) * (*x_iter));
                sum_y += ((*y_iter) * (*y_iter));
                ++x_iter;
                ++y_iter;
            }
        }
    }
};

template <typename TNum, typename TDistance = euclidean_distance_metric<TNum> >
class dbscan_sparse : public dbscan<TNum> {
    // dbscan implementation using sparse arrays and supporting different
    // distance metrics via the TDistance template.
public:
    dbscan_sparse(const TNum* corpus, index_t rows, index_t cols);
    virtual ~dbscan_sparse() {}

protected:
    virtual index_t region_query(index_t vec_i, TNum eps, index_set& result) override;
    std::vector<corpus_vector_t<TNum> > _corpus;
};

template <typename TNum, typename TDistance>
dbscan_sparse<TNum, TDistance>::dbscan_sparse(const TNum* corpus, index_t rows, 
        index_t cols) {
    this->_rows = rows;
    this->_cols = cols;

    for (index_t i=0; i < rows; i++) {
        corpus_vector_t<TNum> vec (cols);
        for (index_t j=0; j < cols; j++) {
            TNum val = corpus[i*cols+j];
            if (val != 0) {
                vec.push_back(j, val);
            }
        }
        this->_corpus.push_back(vec);
    }
}

template <typename TNum, typename TDistance>
index_t dbscan_sparse<TNum, TDistance>::region_query(index_t vec_i, TNum eps, 
        index_set& result)
{
    TDistance _distance_metric(eps);
    
    const corpus_vector_t<TNum>& comparison_vector = _corpus[vec_i];
    for (index_t i=0; i < this->_rows; i++) {
        if (i == vec_i) {
            continue;
        }

        bool included = _distance_metric(this->_corpus[i], comparison_vector);
        if (included) {
            result.insert(i);
        }
    }    

    return result.size();
}

}

#endif
