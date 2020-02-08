/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Multivariate t-Student Distribution                                      *
***************************************************************************/

#ifndef MULTIVARIATETSTUDENT_HPP
#define MULTIVARIATETSTUDENT_HPP

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

/***************************** Multivariate t-Student *****************************/

class MultivariateTStudent
{
public:
    MultivariateTStudent();
    MultivariateTStudent(const MatrixXd &dataset, const double regularization = 0);
    MultivariateTStudent(const RowVectorXd &mu, const MatrixXd &sigma, const double nu, const double regularization = 0);
    MultivariateTStudent(const MultivariateTStudent &other);

    RowVectorXd mu() const { return m_mu; }
    MatrixXd sigma() const { return m_sigma; }
    double nu() const { return m_nu; }
    int dimensions() const { return m_dimensions; }
    bool isWellConditioned() const { return m_wellConditioned; }
    double regularization() const { return m_regularization; }

    void setParameters(const RowVectorXd &mu, const MatrixXd &sigma, const double nu);
    void setRegularization(const double regularization) { m_regularization = regularization; }

    void maximumLikelihoodEstimate(const MatrixXd &dataset);
    void maximumLikelihoodEstimate(const MatrixXd &dataset, const VectorXd &posteriorWeights, const VectorXd &scaleWeights);
    ArrayXd probabilityDensityFunction(const MatrixXd &dataset);
    ArrayXd scaleFunction(const MatrixXd &dataset);
    
    MultivariateTStudent& operator = (const MultivariateTStudent &other);

    friend std::ostream& operator <<(std::ostream &os, const MultivariateTStudent &mvt);

private:
    template <class T>
    class degreesFreedomFunctor
    {
    public:
        degreesFreedomFunctor(const double constant): m_constant(constant) {}
        std::pair<T, T> operator()(const T &nu)
        {
            T fx = std::log(nu / 2.0) - boost::math::digamma(nu / 2.0) + m_constant;
            T dx = (1.0 / nu) - (boost::math::trigamma(nu / 2.0) / 2.0);
            return std::make_pair(fx, dx);
        }
    private:
        double m_constant;
    };

    RowVectorXd m_mu;
    MatrixXd m_sigma;
    MatrixXd m_isigma;
    double   m_dsigma;
    double   m_nu;
    double   m_const;
    double   m_numConst;
    double   m_denConst;
    int      m_dimensions;
    bool     m_wellConditioned;
    double   m_regularization;
    bool     m_shouldComputeSMD;
    VectorXd m_squaredMahalanobisDistance;
};

/***************************** Implementation *****************************/

MultivariateTStudent::MultivariateTStudent()
: m_mu(RowVectorXd()), m_sigma(MatrixXd()), m_isigma(MatrixXd()), m_dsigma(0), m_nu(SVFMM_STUDENT_DEFAULT_NU), m_const(0), m_numConst(0), m_denConst(0), m_dimensions(0), m_wellConditioned(false), m_regularization(0), m_shouldComputeSMD(true), m_squaredMahalanobisDistance(VectorXd())
{
}

MultivariateTStudent::MultivariateTStudent(const MatrixXd &dataset, const double regularization)
: m_regularization(regularization)
{
    maximumLikelihoodEstimate(dataset);
}

MultivariateTStudent::MultivariateTStudent(const RowVectorXd &mu, const MatrixXd &sigma, const double nu, const double regularization)
: m_regularization(regularization)
{
    setParameters(mu, sigma, nu);
}

MultivariateTStudent::MultivariateTStudent(const MultivariateTStudent &other)
: m_mu(other.mu()), m_sigma(other.sigma()), m_nu(other.nu()), m_dimensions(other.dimensions()), m_wellConditioned(other.isWellConditioned()), m_regularization(other.regularization()), m_shouldComputeSMD(true), m_squaredMahalanobisDistance(VectorXd())
{
    m_isigma = m_sigma.inverse();
    m_dsigma = m_sigma.determinant();
    m_const = (m_nu + m_dimensions) / 2.0;
    m_numConst = boost::math::tgamma(m_const) * std::pow(m_dsigma, -0.5);
    m_denConst = std::pow(SVFMM_PI * m_nu, m_dimensions / 2.0) * boost::math::tgamma(m_nu / 2.0);
}

void MultivariateTStudent::setParameters(const RowVectorXd &mu, const MatrixXd &sigma, const double nu)
{
    if (mu.cols() != sigma.rows() || sigma.rows() != sigma.cols() || nu < 0)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent parameters" << std::endl;
        throw std::runtime_error(s.str());
    }

    m_mu = mu;
    m_sigma = sigma;
    m_isigma = sigma.inverse();
    m_dsigma = sigma.determinant();
    m_wellConditioned = sigma.fullPivLu().isInvertible();
    if (!m_wellConditioned && m_regularization != 0)
    {
        m_sigma = sigma + (MatrixXd::Identity(m_sigma.rows(), m_sigma.cols()) * m_regularization);
        m_isigma = m_sigma.inverse();
        m_dsigma = m_sigma.determinant();
        m_wellConditioned = m_sigma.fullPivLu().isInvertible();
    }
    m_dimensions = (int) sigma.cols();
    m_nu = nu;
    m_const = (m_nu + m_dimensions) / 2.0;
    m_numConst = boost::math::tgamma(m_const) * std::pow(m_dsigma, -0.5);
    m_denConst = std::pow(SVFMM_PI * m_nu, m_dimensions / 2.0) * boost::math::tgamma(m_nu / 2.0);
    m_shouldComputeSMD = true;
    
    if (!m_wellConditioned)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Bad conditioned distribution" << std::endl;
        throw std::runtime_error(s.str());
    }
}

void MultivariateTStudent::maximumLikelihoodEstimate(const MatrixXd &dataset)
{
    m_mu = dataset.colwise().mean();
    MatrixXd centered = dataset.rowwise() - m_mu;
    m_sigma = (centered.adjoint() * centered) / (double) (dataset.rows() - 1);
    m_isigma = m_sigma.inverse();
    m_dsigma = m_sigma.determinant();
    m_wellConditioned = m_sigma.fullPivLu().isInvertible();
    if (!m_wellConditioned && m_regularization != 0)
    {
        m_sigma = m_sigma + (MatrixXd::Identity(m_sigma.rows(), m_sigma.cols()) * m_regularization);
        m_isigma = m_sigma.inverse();
        m_dsigma = m_sigma.determinant();
        m_wellConditioned = m_sigma.fullPivLu().isInvertible();
    }
    m_dimensions = (int) m_sigma.cols();
    m_nu = SVFMM_STUDENT_DEFAULT_NU;
    m_const = (m_nu + m_dimensions) / 2.0;
    m_numConst = boost::math::tgamma(m_const) * std::pow(m_dsigma, -0.5);
    m_denConst = std::pow(SVFMM_PI * m_nu, m_dimensions / 2.0) * boost::math::tgamma(m_nu / 2.0);
    m_shouldComputeSMD = true;
    
    if (!m_wellConditioned)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Bad conditioned distribution" << std::endl;
        throw std::runtime_error(s.str());
    }
}

void MultivariateTStudent::maximumLikelihoodEstimate(const MatrixXd &dataset, const VectorXd &posteriorWeights, const VectorXd &scaleWeights)
{
    const VectorXd weights = posteriorWeights.cwiseProduct(scaleWeights);
    double weightsum = weights.sum();
    double posteriorWeightsSum = posteriorWeights.sum();

    m_mu = (weights.transpose() * dataset) / weightsum;
    MatrixXd centered(dataset.rows(), dataset.cols());
#pragma omp parallel for
    for (int i = 0; i < dataset.rows(); ++i)
        centered.row(i) = (dataset.row(i) - m_mu) * sqrt(weights(i));
    m_sigma = (centered.adjoint() * centered) / posteriorWeightsSum;
    m_isigma = m_sigma.inverse();
    m_dsigma = m_sigma.determinant();
    m_wellConditioned = m_sigma.fullPivLu().isInvertible();
    if (!m_wellConditioned && m_regularization != 0)
    {
        m_sigma = m_sigma + (MatrixXd::Identity(m_sigma.rows(), m_sigma.cols()) * m_regularization);
        m_isigma = m_sigma.inverse();
        m_dsigma = m_sigma.determinant();
        m_wellConditioned = m_sigma.fullPivLu().isInvertible();
    }
    m_dimensions = (int) m_sigma.cols();

    double temp = 0;
    for (int i = 0; i < dataset.rows(); ++i)
        temp += (posteriorWeights(i) * (std::log(scaleWeights(i)) - scaleWeights(i)));
    const double constant = 1.0 - std::log((m_nu + m_dimensions) / 2.0) + (temp / posteriorWeightsSum) + boost::math::digamma((m_nu + m_dimensions) / 2.0);
    
    boost::uintmax_t maxIterations = 500;
    m_nu = boost::math::tools::newton_raphson_iterate(degreesFreedomFunctor<double>(constant), m_nu, 0.001, 50.0, static_cast<int>(std::numeric_limits<double>::digits * 0.6), maxIterations);
    m_const = (m_nu + m_dimensions) / 2.0;
    m_numConst = boost::math::tgamma(m_const) * std::pow(m_dsigma, -0.5);
    m_denConst = std::pow(SVFMM_PI * m_nu, m_dimensions / 2.0) * boost::math::tgamma(m_nu / 2.0);
    m_shouldComputeSMD = true;
    
    if (!m_wellConditioned)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Bad conditioned distribution" << std::endl;
        throw std::runtime_error(s.str());
    }
}

ArrayXd MultivariateTStudent::probabilityDensityFunction(const MatrixXd &dataset)
{
    if (!m_wellConditioned)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Bad conditioned distribution" << std::endl;
        throw std::runtime_error(s.str());
    }

    ArrayXd pdf(dataset.rows());
    if (m_shouldComputeSMD)
        m_squaredMahalanobisDistance.resize(dataset.rows());
#pragma omp parallel for
    for (int i = 0; i < dataset.rows(); ++i)
    {
        if (m_shouldComputeSMD)
        {
            RowVectorXd diff = dataset.row(i) - m_mu;
            m_squaredMahalanobisDistance(i) = diff * m_isigma * diff.transpose();
        }
        const double p = m_numConst / (m_denConst * std::pow(1.0 + (m_squaredMahalanobisDistance(i) / m_nu), m_const));
        pdf(i) = std::isnormal(p) && p > SVFMM_ZERO ? p : SVFMM_ZERO;
    }
    m_shouldComputeSMD = false;
    return pdf;
}

ArrayXd MultivariateTStudent::scaleFunction(const MatrixXd &dataset)
{
    if (!m_wellConditioned)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Bad conditioned distribution" << std::endl;
        throw std::runtime_error(s.str());
    }

    ArrayXd values(dataset.rows());
    if (m_shouldComputeSMD)
        m_squaredMahalanobisDistance.resize(dataset.rows());
#pragma omp parallel for
    for (int i = 0; i < dataset.rows(); ++i)
    {
        if (m_shouldComputeSMD)
        {
            RowVectorXd diff = dataset.row(i) - m_mu;
            m_squaredMahalanobisDistance(i) = diff * m_isigma * diff.transpose();
        }
        values(i) = (m_nu + m_dimensions) / (m_nu + m_squaredMahalanobisDistance(i));
    }
    m_shouldComputeSMD = false;
    return values;
}

MultivariateTStudent& MultivariateTStudent::operator = (const MultivariateTStudent &other)
{
    m_mu = other.mu();
    m_sigma = other.sigma();
    m_nu = other.nu();
    m_dimensions = other.dimensions();
    m_wellConditioned = other.isWellConditioned();
    m_regularization = other.regularization();
    
    m_isigma = m_sigma.inverse();
    m_dsigma = m_sigma.determinant();
    m_const = (m_nu + m_dimensions) / 2.0;
    m_numConst = boost::math::tgamma(m_const) * std::pow(m_dsigma, -0.5);
    m_denConst = std::pow(SVFMM_PI * m_nu, m_dimensions / 2.0) * boost::math::tgamma(m_nu / 2.0);
    m_squaredMahalanobisDistance = VectorXd();
    m_shouldComputeSMD = true;
    
    return *this;
}

std::ostream& operator <<(std::ostream &os, const MultivariateTStudent &mvt)
{
    os << "Mean:" << std::endl << mvt.m_mu << std::endl << "Covariance matrix:" << std::endl << mvt.m_sigma << std::endl << "Degrees of freedom:" << std::endl << mvt.m_nu << std::endl;
    return os;
}

#endif