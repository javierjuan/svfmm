/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Spatially Variant Finite Mixture Model Base                              *
***************************************************************************/

#ifndef SPATIALLYVARIANTFINITEMIXTUREMODELBASE_HPP
#define SPATIALLYVARIANTFINITEMIXTUREMODELBASE_HPP

#include <Eigen/Dense>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <memory>
#include "Constants.hpp"
#include "MarkovRandomFields/MarkovRandomField.hpp"

using namespace Eigen;

/***************************** Spatially Variant Finite Mixture Model Base *****************************/

template<typename Distribution, int Dimensions>
class SpatiallyVariantFiniteMixtureModelBase
{
public:
    virtual ~SpatiallyVariantFiniteMixtureModelBase() {}

    std::vector<Distribution> distributions() const { return m_distributions; }
    std::unique_ptr<MarkovRandomField<Dimensions>> const& spatialCoefficients() const { return m_spatialCoefficients; }
    ArrayXXd coefficients() const { return m_spatialCoefficients->coefficients(); }
    int observations() const { return m_observations; }
    int components() const { return m_components; }
    int spectrum() const { return m_spectrum; }
    ArrayXXd jointDensities() const { return m_jointDensities; }
    ArrayXXd posteriorProbabilities() const { return m_posteriorProbabilities; }
    ArrayXd evidenceDensities() const { return m_evidenceDensities; }
    double logLikelihood() const { return m_logLikelihood; }
    std::vector<double> logLikelihoodHistory() const { return m_logLikelihoodHistory; }
    VectorXi labels() const;

    bool isEmpty() const { return m_distributions.size() == 0 || m_spatialCoefficients->isEmpty() || m_components == 0; }
    bool isValid() const { return !isEmpty() && m_distributions.size() == (size_t) m_components && m_spatialCoefficients->classes() == m_components; }
    bool isWellConditioned() const;
    
    void setMixture(const std::vector<Distribution> &distributions, std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients);
    void addComponentToMixture(const Distribution &distribution, const ArrayXd &coefficient);
    void updateComponentOfMixture(const Distribution &distribution, const ArrayXd &coefficient, const int index);
    
    void setDistributions(const std::vector<Distribution> &distributions);
    void setDistribution(const Distribution &distribution, const int index);
    
    void setSpatialCoefficients(std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients);
    void setCoefficients(const ArrayXXd &coefficients) { m_spatialCoefficients->setCoefficients(coefficients); }
    void setCoefficientsPerComponent(const ArrayXd &coefficients, const int index) { m_spatialCoefficients->setClassCoefficients(coefficients, index); }
    void setCoefficientsPerObservation(const ArrayXd &coefficients, const int index) { m_spatialCoefficients->setNodeCoefficients(coefficients, index); }
    
    Distribution  distribution(const int index) const { return m_distributions[index]; }
    Distribution& distribution(const int index)       { return m_distributions[index]; }
    Distribution  operator [] (const int index) const { return m_distributions[index]; }
    Distribution& operator [] (const int index)       { return m_distributions[index]; }
    ArrayXd coefficientsPerComponent(const int index) const { return m_spatialCoefficients->classCoefficients(index); }
    ArrayXd coefficientsPerObservation(const int index) const { return m_spatialCoefficients->nodeCoefficients(index); }
    ArrayXd  operator () (const int index) const { return m_spatialCoefficients->classCoefficients(index); }
    ArrayXd& operator () (const int index)       { return m_spatialCoefficients->classCoefficients(index); }

    virtual void initialization(const MatrixXd &dataset, const int K, const VectorXi &labels, const double regularization = 0) = 0;
    virtual void updateDistributions(const MatrixXd &dataset) = 0;
    virtual void updateParameters(const MatrixXd &dataset) = 0;
    double updateLogLikelihood();

    SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>& operator = (const SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions> &other);
    template<typename, int> friend std::ostream& operator <<(std::ostream &os, const SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions> &model);

protected:
    SpatiallyVariantFiniteMixtureModelBase();
    SpatiallyVariantFiniteMixtureModelBase(std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients);
    SpatiallyVariantFiniteMixtureModelBase(const std::vector<Distribution> &distributions, std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients);
    SpatiallyVariantFiniteMixtureModelBase(const SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions> &other);

    std::vector<Distribution> m_distributions;
    std::unique_ptr<MarkovRandomField<Dimensions>> m_spatialCoefficients;
    int m_observations;
    int m_components;
    int m_spectrum;
    ArrayXXd m_jointDensities;
    ArrayXXd m_posteriorProbabilities;
    ArrayXd m_evidenceDensities;
    std::vector<double> m_logLikelihoodHistory;
    double m_logLikelihood;
};

/***************************** Implementation *****************************/

template<typename Distribution, int Dimensions>
SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>::SpatiallyVariantFiniteMixtureModelBase()
: m_spatialCoefficients(nullptr), m_observations(0), m_components(0), m_spectrum(0), m_jointDensities(ArrayXXd()), m_posteriorProbabilities(ArrayXXd()), m_evidenceDensities(ArrayXd()), m_logLikelihood(-SVFMM_INF)
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    m_distributions.clear();
    m_logLikelihoodHistory.clear();
}

template<typename Distribution, int Dimensions>
SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>::SpatiallyVariantFiniteMixtureModelBase(std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients)
: m_spatialCoefficients(std::move(spatialCoefficients)), m_spectrum(0), m_jointDensities(ArrayXXd()), m_posteriorProbabilities(ArrayXXd()), m_evidenceDensities(ArrayXd()), m_logLikelihood(-SVFMM_INF)
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    m_observations = m_spatialCoefficients->nodes();
    m_components = m_spatialCoefficients->classes();
    m_distributions.clear();
    m_logLikelihoodHistory.clear();
}

template<typename Distribution, int Dimensions>
SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>::SpatiallyVariantFiniteMixtureModelBase(const std::vector<Distribution> &distributions, std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients)
: m_distributions(distributions), m_spatialCoefficients(std::move(spatialCoefficients)), m_spectrum(distributions.front().dimensions()), m_jointDensities(ArrayXXd()), m_posteriorProbabilities(ArrayXXd()), m_evidenceDensities(ArrayXd()), m_logLikelihood(-SVFMM_INF)
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    m_observations = m_spatialCoefficients->nodes();
    m_components = m_spatialCoefficients->classes();
    m_distributions.clear();
    m_logLikelihoodHistory.clear();
    
    if (!isValid())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent mixture model" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<typename Distribution, int Dimensions>
SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>::SpatiallyVariantFiniteMixtureModelBase(const SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions> &other)
: m_distributions(other.distributions()), m_spatialCoefficients(std::move(other.spatialCoefficients()->clone())), m_observations(other.observations()), m_components(other.components()), m_spectrum(other.spectrum()), m_jointDensities(other.jointDensities()), m_posteriorProbabilities(other.posteriorProbabilities()), m_evidenceDensities(other.evidenceDensities()), m_logLikelihood(other.logLikelihood()), m_logLikelihoodHistory(other.logLikelihoodHistory())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    if (!isValid())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent mixture model" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<typename Distribution, int Dimensions>
SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>& SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>::operator = (const SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions> &other)
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    m_distributions.clear();
    m_distributions = other.distributions();
    m_spatialCoefficients = std::move(other.spatialCoefficients()->clone());
    m_observations = other.observations();
    m_components = other.components();
    m_spectrum = other.spectrum();
    m_jointDensities = other.jointDensities();
    m_posteriorProbabilities = other.posteriorProbabilities();
    m_evidenceDensities = other.evidenceDensities();
    m_logLikelihood = other.logLikelihood();
    m_logLikelihoodHistory = other.logLikelihoodHistory();
    
    if (!isValid())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent mixture model" << std::endl;
        throw std::runtime_error(s.str());
    }

    return *this;
}

template<typename Distribution, int Dimensions>
bool SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>::isWellConditioned() const
{
    if (!isValid())
        return false;

    for (int i = 0; i < (int) m_distributions.size(); ++i)
    {
        if (!m_distributions[i].isWellConditioned())
            return false;
    }
    return true;
}

template<typename Distribution, int Dimensions>
void SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>::setMixture(const std::vector<Distribution> &distributions, std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients)
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    m_distributions.clear();
    m_distributions = distributions;
    m_spatialCoefficients = std::move(spatialCoefficients);
    m_observations = m_spatialCoefficients->nodes();
    m_components = m_spatialCoefficients->classes();
    m_spectrum = m_distributions.front().dimensions();
    m_jointDensities.resize(0,0);
    m_posteriorProbabilities.resize(0,0);
    m_evidenceDensities.resize(0);
    m_logLikelihood = -SVFMM_INF;
    m_logLikelihoodHistory.clear();
    
    if (!isValid())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent mixture model" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<typename Distribution, int Dimensions>
void SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>::addComponentToMixture(const Distribution &distribution, const ArrayXd &coefficients)
{
    if (!distribution.isWellConditioned())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Distribution not well conditioned" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    if (distribution.dimensions() != m_spectrum)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Incompatible distribution dimensions" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    ++m_components;
    m_distributions.push_back(distribution);
    m_spatialCoefficients->addClassToMarkovRandomField(coefficients);
}

template<typename Distribution, int Dimensions>
void SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>::updateComponentOfMixture(const Distribution &distribution, const ArrayXd &coefficients, const int index)
{
    if (index < 0 || index >= m_distributions.size() || index >= m_components || index >= m_spatialCoefficients->classes())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Index out of range" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    if (!distribution.isWellConditioned())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Distribution not well conditioned" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    if (distribution.dimensions() != m_spectrum)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Incompatible distribution dimensions" << std::endl;
        throw std::runtime_error(s.str());
    }

    m_distributions[index] = distribution;
    m_spatialCoefficients->setClassCoefficients(coefficients, index);
}

template<typename Distribution, int Dimensions>
void SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>::setDistributions(const std::vector<Distribution> &distributions)
{
    if (m_distributions.size() != distributions.size())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent distributions" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    bool wellConditioned = true;
    for (int i = 0; i < (int) distributions.size(); ++i)
    {
        if (!distributions[i].isWellConditioned())
        {
            wellConditioned = false;
            break;
        }
    }
    if (!wellConditioned)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Distributions are not well conditioned" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    m_distributions = distributions;
    m_spectrum = m_distributions.front().dimensions();
}

template<typename Distribution, int Dimensions>
void SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>::setDistribution(const Distribution &distribution, const int index)
{
    if (index < 0 || index >= m_distributions.size())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Index out of range" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    if (!distribution.isWellConditioned())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Distribution not well conditioned" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    if (distribution.dimensions() != m_spectrum)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Incompatible distribution dimensions" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    m_distributions[index] = distribution;
    m_spectrum = m_distributions.front().dimensions();
}

template<typename Distribution, int Dimensions>
void SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>::setSpatialCoefficients(std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients)
{
    if (!m_spatialCoefficients->isCompatible(*spatialCoefficients))
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent Markov Random Field for the spatial coefficients" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    m_spatialCoefficients = std::move(spatialCoefficients);
}

template<typename Distribution, int Dimensions>
void SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>::initialization(const MatrixXd &dataset, const int K, const VectorXi &labels, const double regularization)
{
    if (K < 2)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Invalid number of classes (K must be > 1)" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    m_components = K;
    m_observations = (int) dataset.rows();
    m_spectrum = (int) dataset.cols();
    
    MatrixXd centroids(m_components, m_spectrum);
#pragma omp parallel for
    for (int j = 0; j < m_components; ++j)
    {
        int k = 0;
        RowVectorXd centroid = RowVectorXd::Zero(m_spectrum);
        for (int i = 0; i < m_observations; ++i)
        {
            if (labels(i) == j)
            {
                centroid += dataset.row(i);
                ++k;
            }
        }
        centroids.row(j) = centroid / k;
    }
    
    ArrayXXd initialProbabilities(m_observations, m_components);
#pragma omp parallel for
    for (int i = 0; i < m_observations; ++i)
    {
        Eigen::RowVectorXd p(m_components);
        for (int j = 0; j < m_components; ++j)
        {
            const double d = (dataset.row(i) - centroids.row(j)).norm();
            p(j) = 1.0 / (d == 0 ? SVFMM_EPS : d);
        }
        p /= p.sum();
        initialProbabilities.row(i) = p;
    }
    m_spatialCoefficients->initialization(initialProbabilities);

    m_posteriorProbabilities = initialProbabilities;
    m_jointDensities = ArrayXXd::Zero(m_observations, m_components);
    m_evidenceDensities = ArrayXd::Zero(m_observations);
}

template<typename Distribution, int Dimensions>
double SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>::updateLogLikelihood()
{
    ArrayXd logLikelihood(m_observations);
#pragma omp parallel for
    for (int i = 0; i < m_observations; ++i)
    {
        double ll = 0;
        for (int j = 0; j < m_components; ++j)
        {
            ll += m_posteriorProbabilities(i, j) * (std::log(m_jointDensities(i, j)));
        }
        logLikelihood(i) = ll;
    }
    m_logLikelihood = logLikelihood.sum() + m_spatialCoefficients->logLikelihood();
    m_logLikelihoodHistory.push_back(m_logLikelihood);
    
    return m_logLikelihood;
}

template<typename Distribution, int Dimensions>
VectorXi SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>::labels() const
{
    VectorXi L(m_posteriorProbabilities.rows());
#pragma omp parallel for
    for (int i = 0; i < m_posteriorProbabilities.rows(); ++i)
        m_posteriorProbabilities.row(i).maxCoeff(&L[i]);
    return L;
}

template<typename Distribution, int Dimensions>
std::ostream& operator <<(std::ostream &os, const SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions> &model)
{
    os << "Spatially Variant Finite mixture model" << std::endl << "######################################" << std::endl << std::endl;
    for (int i = 0; i < model.components(); ++i)
        os << "Component: " << i << std::endl << "------------" << std::endl << model[i] << std::endl;
    return os;
}

#endif