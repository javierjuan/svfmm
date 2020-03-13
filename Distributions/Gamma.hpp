/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Gamma Distribution                                                       *
***************************************************************************/

#ifndef GAMMA_HPP
#define GAMMA_HPP

#include <boost/math/tools/roots.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cmath>
#include <omp.h>
#include "../Constants.hpp"

using namespace Eigen;

/***************************** Gamma *****************************/

class Gamma
{
public:
    Gamma();
    Gamma(const VectorXd &dataset, const double regularization = 0);
    Gamma(const double k, const double theta);
    Gamma(const Gamma &other);
    
    double k() const { return m_k; }
    double theta() const { return m_theta; }
    int dimensions() const { return 1; }
    bool isWellConditioned() const { return true; }
    
    void setParameters(const double k, const double theta);
    
    void maximumLikelihoodEstimate(const ArrayXd &dataset);
    void maximumLikelihoodEstimate(const ArrayXd &dataset, const ArrayXd &weights);
    ArrayXd probabilityDensityFunction(const VectorXd &dataset) const;
    
    Gamma& operator = (const Gamma &other);
    
    friend std::ostream& operator <<(std::ostream &os, const Gamma &g);
    
private:
    template <class T>
    class kFunctor
    {
    public:
        kFunctor(const double constant): m_constant(constant) {}
        std::pair<T, T> operator()(const T &k)
        {
            T fx = std::log(k) - boost::math::digamma(k) + m_constant;
            T dx = (1.0 / k) - boost::math::trigamma(k);
            return std::make_pair(fx, dx);
        }
    private:
        double m_constant;
    };

    double m_k;
    double m_theta;
    double m_const;
};

/***************************** Implementation *****************************/

Gamma::Gamma()
: m_k(0), m_theta(0), m_const(0)
{
}

Gamma::Gamma(const VectorXd &dataset,  const double regularization)
{
    maximumLikelihoodEstimate(dataset);
}

Gamma::Gamma(const double k, const double theta)
{
    setParameters(k, theta);
}

Gamma::Gamma(const Gamma &other)
: m_k(other.k()), m_theta(other.theta())
{
    m_const = 1.0 / (double) (boost::math::tgamma(m_k) * std::pow(m_theta, m_k));
}

void Gamma::setParameters(const double k, const double theta)
{
    m_k = k;
    m_theta = theta;
    
    m_const = 1.0 / (double) (boost::math::tgamma(m_k) * std::pow(m_theta, m_k));
}

void Gamma::maximumLikelihoodEstimate(const ArrayXd &dataset)
{
    const double averageDataset = dataset.mean();
    const double constant = dataset.log().mean() - std::log(averageDataset);
    boost::uintmax_t maxIterations = 500;
    m_k = boost::math::tools::newton_raphson_iterate(kFunctor<double>(constant), m_k, 0.001, 50.0, static_cast<int>(std::numeric_limits<double>::digits * 0.6), maxIterations);
    m_theta = averageDataset / m_k;
    
    m_const = 1.0 / (double) (boost::math::tgamma(m_k) * std::pow(m_theta, m_k));
}

void Gamma::maximumLikelihoodEstimate(const ArrayXd &dataset, const ArrayXd &weights)
{
    double weightsum = weights.sum();
    const double normWeightedDataset = (weights * dataset).sum() / weightsum;
    const double constant = ((dataset.log() * weights).sum() / weightsum) - std::log(normWeightedDataset);
    boost::uintmax_t maxIterations = 500;
    m_k = boost::math::tools::newton_raphson_iterate(kFunctor<double>(constant), m_k, 0.001, 50.0, static_cast<int>(std::numeric_limits<double>::digits * 0.6), maxIterations);
    m_theta = normWeightedDataset / m_k;
    
    m_const = 1.0 / (double) (boost::math::tgamma(m_k) * std::pow(m_theta, m_k));
}

ArrayXd Gamma::probabilityDensityFunction(const VectorXd &dataset) const
{
    ArrayXd pdf(dataset.size());
#pragma omp parallel for
    for (int i = 0; i < dataset.size(); ++i)
    {
        const double p = m_const * std::pow(dataset(i), (m_k - 1.0)) * std::exp(-dataset(i) / m_theta);
        pdf(i) = std::isnormal(p) && p > SVFMM_ZERO ? p : SVFMM_ZERO;
    }
    return pdf;
}

Gamma& Gamma::operator = (const Gamma &other)
{
    m_k = other.k();
    m_theta = other.theta();
    
    m_const = 1.0 / (double) (boost::math::tgamma(m_k) * std::pow(m_theta, m_k));
    
    return *this;
}

std::ostream& operator <<(std::ostream &os, const Gamma &g)
{
    os << "k:" << std::endl << g.m_k << std::endl << "Theta:" << std::endl << g.m_theta << std::endl;
    return os;
}

#endif