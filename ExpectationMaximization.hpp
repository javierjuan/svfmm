/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Expectation Maximization Algorithm                                       *
***************************************************************************/

#ifndef EXPECTATIONMAXIMIZATION_HPP
#define EXPECTATIONMAXIMIZATION_HPP

#include <Eigen/Dense>
#include <EigenCImg.hpp>
#include <KMeans.hpp>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <omp.h>
#include "SpatiallyVariantFiniteMixtureModel.hpp"
#include "Constants.hpp"

using namespace cimg_library;
using namespace Eigen;

/***************************** Expectation-Maximization *****************************/

template<typename Distribution, int Dimensions>
class ExpectationMaximization
{
public:
    ExpectationMaximization();
    ExpectationMaximization(const int maxIterations, const double threshold, const double regularization = 0, const bool verbose = false);

    int maxIterations() const { return m_maxIterations; }
    int iterations() const { return m_iteration; }
    double threshold() const { return m_threshold; }
    double regularization() const { return m_regularization; }
    bool verbose() const { return m_verbose; }

    void setMaxIterations(const int maxIterations) { m_maxIterations = maxIterations; }
    void setThreshold(const double threshold) { m_threshold = threshold; }
    void setVerbose(const bool verbose) { m_verbose = verbose; }

    void learn(std::shared_ptr<SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions>> model, const CImg<double> &image, const CImg<bool> &mask, const int K, const CImg<int> &initialLabels = CImg<int>());
    void learn(std::shared_ptr<SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions>> model, const MatrixXd &dataset, const int K, const VectorXi &initialLabels = VectorXi());

private:
    ExpectationMaximization(const ExpectationMaximization &other);
    ExpectationMaximization& operator=(const ExpectationMaximization &other);
    
    void algorithm(const MatrixXd &dataset, int K);
    void initialization(const MatrixXd &dataset, const int K, const VectorXi &labels);
    void expectation(const MatrixXd &dataset);
    void maximization(const MatrixXd &dataset);
    bool convergence();

    std::shared_ptr<SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions>> m_model;
    double m_ratio;
    int m_maxIterations;
    int m_iteration;
    double m_threshold;
    double m_regularization;
    bool m_verbose;
};

/***************************** Implementation *****************************/
template<typename Distribution, int Dimensions>
ExpectationMaximization<Distribution, Dimensions>::ExpectationMaximization()
: m_model(nullptr), m_ratio(0), m_maxIterations(100), m_iteration(0), m_threshold(0.001), m_regularization(0), m_verbose(false)
{
}

template<typename Distribution, int Dimensions>
ExpectationMaximization<Distribution, Dimensions>::ExpectationMaximization(const int maxIterations, const double threshold, const double regularization, const bool verbose)
: m_model(nullptr), m_ratio(0), m_maxIterations(maxIterations), m_iteration(0), m_threshold(threshold), m_regularization(regularization), m_verbose(verbose)
{
    if (maxIterations < 0)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Invalid maximum number of iterations (must be >= 0)" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    if (threshold <= 0)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Invalid threshold (must be > 0)" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<typename Distribution, int Dimensions>
void ExpectationMaximization<Distribution, Dimensions>::learn(std::shared_ptr<SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions>> model, const CImg<double> &image, const CImg<bool> &mask, const int K, const CImg<int> &initialLabels)
{
    const MatrixXd dataset = EigenCImg::toEigen(image, mask);

    if (initialLabels.is_empty())
    {
        if (m_verbose)
        {
            std::cout << "Initializing with K-Means..." << std::flush;
        }
        
        KMeans kmeans(1000);
        kmeans.compute(dataset, K, KMeans::MTSPP);
        
        if (m_verbose)
        {
            std::cout << "DONE!" << std::endl << std::endl;
        }
        
        learn(model, dataset, K, kmeans.labels());
    }
    else
        learn(model, dataset, K, EigenCImg::toEigen(initialLabels, mask));
}

template<typename Distribution, int Dimensions>
void ExpectationMaximization<Distribution, Dimensions>::learn(std::shared_ptr<SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions>> model, const MatrixXd &dataset, const int K, const VectorXi &initialLabels)
{
    if (K < 2)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Invalid number of classes (K must be > 1)" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    m_model = model;
    initialization(dataset, K, initialLabels);
    algorithm(dataset, K);
}

template<typename Distribution, int Dimensions>
void ExpectationMaximization<Distribution, Dimensions>::algorithm(const MatrixXd &dataset, int K)
{
    if (m_verbose)
    {
        std::cout.precision(2);
        std::cout << "Iteration\t\tMax iterations\t\tLog-likelihood\t\tRatio\t\t\tThreshold" << std::endl;
        std::cout << "---------------------------------------------------------------------------------------------------------" << std::endl;
    }
    for (m_iteration = 0; m_iteration < m_maxIterations; ++m_iteration)
    {
        try
        {
            expectation(dataset);
            
            if (convergence())
            {
                if (m_verbose)
                {
                    std::cout << "Converged!" << std::endl;
                }
                break;
            }

            if (m_verbose)
            {
                std::cout << m_iteration << "\t\t\t" << m_maxIterations << "\t\t\t" << std::scientific << m_model->logLikelihood() << "\t\t" << m_ratio << "\t\t" << m_threshold << std::endl;
            }

            maximization(dataset);
        }
        catch (const std::exception &e)
        {
            std::stringstream s;
            s << "In function " << __PRETTY_FUNCTION__ << " ==> Iteration: " << m_iteration << std::endl << e.what();
            throw std::runtime_error(s.str());
        }
    }
    
    if (m_verbose && m_iteration == m_maxIterations)
    {
        std::cout << "Maximum number of iterations reached!" << std::endl;
    }
}

template<typename Distribution, int Dimensions>
void ExpectationMaximization<Distribution, Dimensions>::initialization(const MatrixXd &dataset, const int K, const VectorXi &labels)
{
    if (dataset.rows() != labels.rows())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent initial labels with size of dataset" << std::endl;
        throw std::runtime_error(s.str());
    }

    VectorXi labelsSafe = labels;
    if (labels.minCoeff() != 0)
        labelsSafe = labels.array() - labels.minCoeff();

    ArrayXi classPresence = ArrayXi::Zero(K);
    for (int i = 0; i < labelsSafe.size(); ++i)
    {
        if (labelsSafe(i) < 0 || labelsSafe(i) >= K)
        {
            std::stringstream s;
            s << "In function " << __PRETTY_FUNCTION__ << " ==> Incorrect initial labels. Found values < 0 or >= K" << std::endl;
            throw std::runtime_error(s.str());
        }

        ++classPresence(labelsSafe(i));
    }
    if (!(classPresence > 0).all())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Incorrect initial labels. Some class is not present" << std::endl;
        throw std::runtime_error(s.str());
    }

    try
    {
        m_model->initialization(dataset, K, labelsSafe, m_regularization);
    }
    catch (const std::exception &e)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << std::endl << e.what();
        throw std::runtime_error(s.str());
    }
}

template<typename Distribution, int Dimensions>
void ExpectationMaximization<Distribution, Dimensions>::expectation(const MatrixXd &dataset)
{
    try
    {
        m_model->updateDistributions(dataset);
    }
    catch (const std::exception &e)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << std::endl << e.what();
        throw std::runtime_error(s.str());
    }
}

template<typename Distribution, int Dimensions>
void ExpectationMaximization<Distribution, Dimensions>::maximization(const MatrixXd &dataset)
{
    try
    {
        m_model->updateParameters(dataset);
    }
    catch (const std::exception &e)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << std::endl << e.what();
        throw std::runtime_error(s.str());
    }
}

template<typename Distribution, int Dimensions>
bool ExpectationMaximization<Distribution, Dimensions>::convergence()
{
    if (!m_model->isWellConditioned())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Bad conditioned model" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    const double oldLogLikelihood = m_model->logLikelihood();
    const double newLogLikelihood = m_model->updateLogLikelihood();

    const double logLikelihoodDiff = newLogLikelihood - oldLogLikelihood;
    m_ratio = logLikelihoodDiff / std::fabs(newLogLikelihood);
    return (logLikelihoodDiff >= 0) && (m_ratio < m_threshold);
}

#endif