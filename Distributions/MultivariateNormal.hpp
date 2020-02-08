/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Multivariate Normal Distribution                                         *
***************************************************************************/

#ifndef MULTIVARIATENORMAL_HPP
#define MULTIVARIATENORMAL_HPP

#include <Eigen/Dense>
#include <Eigen/LU>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cmath>
#include <omp.h>
#include "../Constants.hpp"

using namespace Eigen;

/***************************** Multivariate Normal *****************************/

class MultivariateNormal
{
public:
    MultivariateNormal();
    MultivariateNormal(const MatrixXd &dataset, const double regularization = 0);
    MultivariateNormal(const RowVectorXd &mu, const MatrixXd &sigma, const double regularization = 0);
    MultivariateNormal(const MultivariateNormal &other);
    
    RowVectorXd mu() const { return m_mu; }
    MatrixXd sigma() const { return m_sigma; }
    int dimensions() const { return m_dimensions; }
    bool isWellConditioned() const { return m_wellConditioned; }
    double regularization() const { return m_regularization; }

    void setParameters(const RowVectorXd &mu, const MatrixXd &sigma);
    void setRegularization(const double regularization) { m_regularization = regularization; }
    
    void maximumLikelihoodEstimate(const MatrixXd &dataset);
    void maximumLikelihoodEstimate(const MatrixXd &dataset, const VectorXd &weights);
    ArrayXd probabilityDensityFunction(const MatrixXd &dataset) const;
    
    MultivariateNormal& operator = (const MultivariateNormal &other);
    
    friend std::ostream& operator <<(std::ostream &os, const MultivariateNormal &mvn);
    
private:
    RowVectorXd m_mu;
    MatrixXd m_sigma;
    MatrixXd m_isigma;
    double   m_dsigma;
    double   m_const;
    int      m_dimensions;
    bool     m_wellConditioned;
    double   m_regularization;
};

/***************************** Implementation *****************************/

MultivariateNormal::MultivariateNormal()
: m_mu(RowVectorXd()), m_sigma(MatrixXd()), m_isigma(MatrixXd()), m_dsigma(0), m_const(0), m_dimensions(0), m_wellConditioned(false), m_regularization(0)
{
}

MultivariateNormal::MultivariateNormal(const MatrixXd &dataset, const double regularization)
: m_regularization(regularization)
{
    maximumLikelihoodEstimate(dataset);
}

MultivariateNormal::MultivariateNormal(const RowVectorXd &mu, const MatrixXd &sigma, const double regularization)
: m_regularization(regularization)
{
    setParameters(mu, sigma);
}

MultivariateNormal::MultivariateNormal(const MultivariateNormal &other)
: m_mu(other.mu()), m_sigma(other.sigma()), m_dimensions(other.dimensions()), m_wellConditioned(other.isWellConditioned()), m_regularization(other.regularization())
{
    m_isigma = m_sigma.inverse();
    m_dsigma = m_sigma.determinant();
    m_const = std::pow(SVFMM_2PI, (-m_dimensions / 2.0)) * std::pow(m_dsigma, -0.5);
}

void MultivariateNormal::setParameters(const RowVectorXd &mu, const MatrixXd &sigma)
{
    if (mu.cols() != sigma.rows() || sigma.rows() != sigma.cols())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent parameters" << std::endl;
        throw std::runtime_error(s.str());
    }

    m_mu = mu;
    m_sigma = sigma;
    m_isigma = m_sigma.inverse();
    m_dsigma = m_sigma.determinant();
    m_wellConditioned = m_sigma.fullPivLu().isInvertible();
    if (!m_wellConditioned && m_regularization != 0)
    {
        m_sigma = sigma + (MatrixXd::Identity(m_sigma.rows(), m_sigma.cols()) * m_regularization);
        m_isigma = m_sigma.inverse();
        m_dsigma = m_sigma.determinant();
        m_wellConditioned = m_sigma.fullPivLu().isInvertible();
    }
    m_dimensions = (int) m_sigma.cols();
    m_const = std::pow(SVFMM_2PI, (-m_dimensions / 2.0)) * std::pow(m_dsigma, -0.5);
    
    if (!m_wellConditioned)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Bad conditioned distribution" << std::endl;
        throw std::runtime_error(s.str());
    }
}

void MultivariateNormal::maximumLikelihoodEstimate(const MatrixXd &dataset)
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
    m_const = std::pow(SVFMM_2PI, (-m_dimensions / 2.0)) * std::pow(m_dsigma, -0.5);
    
    if (!m_wellConditioned)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Bad conditioned distribution" << std::endl;
        throw std::runtime_error(s.str());
    }
}

void MultivariateNormal::maximumLikelihoodEstimate(const MatrixXd &dataset, const VectorXd &weights)
{
    double weightsum = weights.sum();
    m_mu = (weights.transpose() * dataset) / weightsum;
    MatrixXd centered(dataset.rows(), dataset.cols());
#pragma omp parallel for
    for (int i = 0; i < dataset.rows(); i++)
        centered.row(i) = (dataset.row(i) - m_mu) * sqrt(weights(i));
    m_sigma = (centered.adjoint() * centered) / weightsum;
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
    m_const = std::pow(SVFMM_2PI, (-m_dimensions / 2.0)) * std::pow(m_dsigma, -0.5);
    
    if (!m_wellConditioned)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Bad conditioned distribution" << std::endl;
        throw std::runtime_error(s.str());
    }
}

ArrayXd MultivariateNormal::probabilityDensityFunction(const MatrixXd &dataset) const
{
    if (!m_wellConditioned)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Bad conditioned distribution" << std::endl;
        throw std::runtime_error(s.str());
    }

    ArrayXd pdf(dataset.rows());
#pragma omp parallel for
    for (int i = 0; i < dataset.rows(); ++i)
    {
        const RowVectorXd diff = dataset.row(i) - m_mu;
        const double p = m_const * std::exp(-0.5 * diff * m_isigma * diff.transpose());
        pdf(i) = std::isnormal(p) && p > SVFMM_ZERO ? p : SVFMM_ZERO;
    }
    return pdf;
}

MultivariateNormal& MultivariateNormal::operator = (const MultivariateNormal &other)
{
    m_mu = other.mu();
    m_sigma = other.sigma();
    m_dimensions = other.dimensions();
    m_wellConditioned = other.isWellConditioned();
    m_regularization = other.regularization();
    
    m_isigma = m_sigma.inverse();
    m_dsigma = m_sigma.determinant();
    m_const = std::pow(SVFMM_2PI, (-m_dimensions / 2.0)) * std::pow(m_dsigma, -0.5);
    
    return *this;
}

std::ostream& operator <<(std::ostream &os, const MultivariateNormal &mvn)
{
    os << "Mean:" << std::endl << mvn.m_mu << std::endl << "Covariance matrix:" << std::endl << mvn.m_sigma << std::endl;
    return os;
}

#endif