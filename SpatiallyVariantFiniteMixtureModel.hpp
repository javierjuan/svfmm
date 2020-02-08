/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Spatially Variant Finite Mixture Model                                   *
***************************************************************************/

#ifndef SPATIALLYVARIANTFINITEMIXTUREMODEL_HPP
#define SPATIALLYVARIANTFINITEMIXTUREMODEL_HPP

#include <stdexcept>
#include "SpatiallyVariantFiniteMixtureModelBase.hpp"
#include "Distributions/MultivariateTStudent.hpp"
#include "Distributions/MultivariateNormal.hpp"
#include "Distributions/Gamma.hpp"

/***************************** Spatially Variant Finite Mixture Model *****************************/

template<typename Distribution, int Dimensions>
class SpatiallyVariantFiniteMixtureModel : public SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>
{
public:
    SpatiallyVariantFiniteMixtureModel();
    SpatiallyVariantFiniteMixtureModel(std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients);
    SpatiallyVariantFiniteMixtureModel(const std::vector<Distribution> &distributions, std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients);
    SpatiallyVariantFiniteMixtureModel(const SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions> &other);
    
    SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions> & operator = (const SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions> &other);

    void initialization(const MatrixXd &dataset, const int K, const VectorXi &labels, const double regularization = 0);
    void updateDistributions(const MatrixXd &dataset);
    void updateParameters(const MatrixXd &dataset);
};

template<typename Distribution, int Dimensions>
SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions>::SpatiallyVariantFiniteMixtureModel()
: SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>()
{
}

template<typename Distribution, int Dimensions>
SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions>::SpatiallyVariantFiniteMixtureModel(std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients)
: SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>(std::move(spatialCoefficients))
{
}

template<typename Distribution, int Dimensions>
SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions>::SpatiallyVariantFiniteMixtureModel(const std::vector<Distribution> &distributions, std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients)
: SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>(distributions, std::move(spatialCoefficients))
{
}

template<typename Distribution, int Dimensions>
SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions>::SpatiallyVariantFiniteMixtureModel(const SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions> &other)
: SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>(other)
{
}

template<typename Distribution, int Dimensions>
SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions>& SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions>::operator = (const SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions> &other)
{
    SpatiallyVariantFiniteMixtureModelBase<Distribution, Dimensions>::operator=(other);

    return *this;
}

template<typename Distribution, int Dimensions>
void SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions>::initialization(const MatrixXd &dataset, const int K, const VectorXi &labels, const double regularization)
{
    std::stringstream s;
    s << "In function " << __PRETTY_FUNCTION__ << " ==> Initialization must be specialized for the each distribution type." << std::endl;
    throw std::runtime_error(s.str());
}

template<typename Distribution, int Dimensions>
void SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions>::updateDistributions(const MatrixXd &dataset)
{
    std::stringstream s;
    s << "In function " << __PRETTY_FUNCTION__ << " ==> Distribution update must be specialized for the each distribution type." << std::endl;
    throw std::runtime_error(s.str());
}

template<typename Distribution, int Dimensions>
void SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions>::updateParameters(const MatrixXd &dataset)
{
    std::stringstream s;
    s << "In function " << __PRETTY_FUNCTION__ << " ==> Parameter update must be specialized for the each distribution type." << std::endl;
    throw std::runtime_error(s.str());
}

/***************************** Template Specialization Multivariate Normal *****************************/

template<int Dimensions>
class SpatiallyVariantFiniteMixtureModel<MultivariateNormal, Dimensions> : public SpatiallyVariantFiniteMixtureModelBase<MultivariateNormal, Dimensions>
{
public:
    SpatiallyVariantFiniteMixtureModel();
    SpatiallyVariantFiniteMixtureModel(std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients);
    SpatiallyVariantFiniteMixtureModel(const std::vector<MultivariateNormal> &distributions, std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients);
    SpatiallyVariantFiniteMixtureModel(const SpatiallyVariantFiniteMixtureModel<MultivariateNormal, Dimensions> &other);
    
    SpatiallyVariantFiniteMixtureModel<MultivariateNormal, Dimensions> & operator = (const SpatiallyVariantFiniteMixtureModel<MultivariateNormal, Dimensions> &other);
    
    void initialization(const MatrixXd &dataset, const int K, const VectorXi &labels, const double regularization = 0);
    void updateDistributions(const MatrixXd &dataset);
    void updateParameters(const MatrixXd &dataset);
};

template<int Dimensions>
SpatiallyVariantFiniteMixtureModel<MultivariateNormal, Dimensions>::SpatiallyVariantFiniteMixtureModel()
: SpatiallyVariantFiniteMixtureModelBase<MultivariateNormal, Dimensions>()
{
}

template<int Dimensions>
SpatiallyVariantFiniteMixtureModel<MultivariateNormal, Dimensions>::SpatiallyVariantFiniteMixtureModel(std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients)
: SpatiallyVariantFiniteMixtureModelBase<MultivariateNormal, Dimensions>(std::move(spatialCoefficients))
{
}

template<int Dimensions>
SpatiallyVariantFiniteMixtureModel<MultivariateNormal, Dimensions>::SpatiallyVariantFiniteMixtureModel(const std::vector<MultivariateNormal> &distributions, std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients)
: SpatiallyVariantFiniteMixtureModelBase<MultivariateNormal, Dimensions>(distributions, std::move(spatialCoefficients))
{
}

template<int Dimensions>
SpatiallyVariantFiniteMixtureModel<MultivariateNormal, Dimensions>::SpatiallyVariantFiniteMixtureModel(const SpatiallyVariantFiniteMixtureModel<MultivariateNormal, Dimensions> &other)
: SpatiallyVariantFiniteMixtureModelBase<MultivariateNormal, Dimensions>(other)
{
}

template<int Dimensions>
SpatiallyVariantFiniteMixtureModel<MultivariateNormal, Dimensions>& SpatiallyVariantFiniteMixtureModel<MultivariateNormal, Dimensions>::operator = (const SpatiallyVariantFiniteMixtureModel<MultivariateNormal, Dimensions> &other)
{
    SpatiallyVariantFiniteMixtureModelBase<MultivariateNormal, Dimensions>::operator=(other);

    return *this;
}

template<int Dimensions>
void SpatiallyVariantFiniteMixtureModel<MultivariateNormal, Dimensions>::initialization(const MatrixXd &dataset, const int K, const VectorXi &labels, const double regularization)
{
    SpatiallyVariantFiniteMixtureModelBase<MultivariateNormal, Dimensions>::initialization(dataset, K, labels);
    
    for (int j = 0; j < this->m_components; ++j)
    {
        int n = 0;
        MatrixXd classData = MatrixXd::Zero(this->m_observations, this->m_spectrum);
        for (int i = 0; i < this->m_observations; ++i)
        {
            if (labels(i) == j)
            {
                classData.row(n) = dataset.row(i);
                ++n;
            }
        }
        classData.conservativeResize(n, this->m_spectrum);
        this->m_distributions.push_back(MultivariateNormal(classData, regularization));
    }
    
    if (!this->isWellConditioned())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Bad conditioned model at initialization" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
void SpatiallyVariantFiniteMixtureModel<MultivariateNormal, Dimensions>::updateDistributions(const MatrixXd &dataset)
{
    for (int j = 0; j < this->m_components; ++j)
    {
        try
        {
            this->m_jointDensities.col(j) = this->m_distributions[j].probabilityDensityFunction(dataset) * this->m_spatialCoefficients->classCoefficients(j);
        }
        catch (const std::exception &e)
        {
            std::stringstream s;
            s << "In function " << __PRETTY_FUNCTION__ << " ==> Component: " << j << std::endl << e.what();
            throw std::runtime_error(s.str());
        }
    }

#pragma omp parallel for
    for (int i = 0; i < this->m_observations; ++i)
    {
        const ArrayXd row = this->m_jointDensities.row(i);
        this->m_evidenceDensities(i) = row.sum();
        this->m_posteriorProbabilities.row(i) = row / this->m_evidenceDensities(i);
    }
    
    this->m_spatialCoefficients->updateDistributions();
}

template<int Dimensions>
void SpatiallyVariantFiniteMixtureModel<MultivariateNormal, Dimensions>::updateParameters(const MatrixXd &dataset)
{
    for (int j = 0; j < this->m_components; ++j)
    {
        try
        {
            this->m_distributions[j].maximumLikelihoodEstimate(dataset, this->m_posteriorProbabilities.col(j));
        }
        catch (const std::exception &e)
        {
            std::stringstream s;
            s << "In function " << __PRETTY_FUNCTION__ << " ==> Component: " << j << std::endl << e.what();
            throw std::runtime_error(s.str());
        }
    }

    this->m_spatialCoefficients->updateParameters(this->m_posteriorProbabilities);
}

/***************************** Template Specialization Multivariate t-Student *****************************/

template<int Dimensions>
class SpatiallyVariantFiniteMixtureModel<MultivariateTStudent, Dimensions> : public SpatiallyVariantFiniteMixtureModelBase<MultivariateTStudent, Dimensions>
{
public:
    SpatiallyVariantFiniteMixtureModel();
    SpatiallyVariantFiniteMixtureModel(std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients);
    SpatiallyVariantFiniteMixtureModel(const std::vector<MultivariateTStudent> &distributions, std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients);
    SpatiallyVariantFiniteMixtureModel(const SpatiallyVariantFiniteMixtureModel<MultivariateTStudent, Dimensions> &other);

    ArrayXXd scaleWeights() const { return m_scaleWeights; }

    SpatiallyVariantFiniteMixtureModel<MultivariateTStudent, Dimensions> & operator = (const SpatiallyVariantFiniteMixtureModel<MultivariateTStudent, Dimensions> &other);

    void initialization(const MatrixXd &dataset, const int K, const VectorXi &labels, const double regularization = 0);
    void updateDistributions(const MatrixXd &dataset);
    void updateParameters(const MatrixXd &dataset);
    
protected:
    ArrayXXd m_scaleWeights;
};

template<int Dimensions>
SpatiallyVariantFiniteMixtureModel<MultivariateTStudent, Dimensions>::SpatiallyVariantFiniteMixtureModel()
: SpatiallyVariantFiniteMixtureModelBase<MultivariateTStudent, Dimensions>(), m_scaleWeights(ArrayXXd())
{
}

template<int Dimensions>
SpatiallyVariantFiniteMixtureModel<MultivariateTStudent, Dimensions>::SpatiallyVariantFiniteMixtureModel(std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients)
: SpatiallyVariantFiniteMixtureModelBase<MultivariateTStudent, Dimensions>(std::move(spatialCoefficients)), m_scaleWeights(ArrayXXd())
{
}

template<int Dimensions>
SpatiallyVariantFiniteMixtureModel<MultivariateTStudent, Dimensions>::SpatiallyVariantFiniteMixtureModel(const std::vector<MultivariateTStudent> &distributions, std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients)
: SpatiallyVariantFiniteMixtureModelBase<MultivariateTStudent, Dimensions>(distributions, std::move(spatialCoefficients)), m_scaleWeights(ArrayXXd())
{
}

template<int Dimensions>
SpatiallyVariantFiniteMixtureModel<MultivariateTStudent, Dimensions>::SpatiallyVariantFiniteMixtureModel(const SpatiallyVariantFiniteMixtureModel<MultivariateTStudent, Dimensions> &other)
: SpatiallyVariantFiniteMixtureModelBase<MultivariateTStudent, Dimensions>(other), m_scaleWeights(other.scaleWeights())
{
}

template<int Dimensions>
SpatiallyVariantFiniteMixtureModel<MultivariateTStudent, Dimensions>& SpatiallyVariantFiniteMixtureModel<MultivariateTStudent, Dimensions>::operator = (const SpatiallyVariantFiniteMixtureModel<MultivariateTStudent, Dimensions> &other)
{
    SpatiallyVariantFiniteMixtureModelBase<MultivariateTStudent, Dimensions>::operator=(other);
    m_scaleWeights = other.scaleWeights();

    return *this;
}

template<int Dimensions>
void SpatiallyVariantFiniteMixtureModel<MultivariateTStudent, Dimensions>::initialization(const MatrixXd &dataset, const int K, const VectorXi &labels, const double regularization)
{
    SpatiallyVariantFiniteMixtureModelBase<MultivariateTStudent, Dimensions>::initialization(dataset, K, labels);
    m_scaleWeights = ArrayXXd::Zero(this->m_observations, this->m_components);
    
    for (int j = 0; j < this->m_components; ++j)
    {
        int n = 0;
        MatrixXd classData = MatrixXd::Zero(this->m_observations, this->m_spectrum);
        for (int i = 0; i < this->m_observations; ++i)
        {
            if (labels(i) == j)
            {
                classData.row(n) = dataset.row(i);
                ++n;
            }
        }
        classData.conservativeResize(n, this->m_spectrum);
        this->m_distributions.push_back(MultivariateTStudent(classData, regularization));
    }
    
    if (!this->isWellConditioned())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Bad conditioned model at initialization" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
void SpatiallyVariantFiniteMixtureModel<MultivariateTStudent, Dimensions>::updateDistributions(const MatrixXd &dataset)
{
    for (int j = 0; j < this->m_components; ++j)
    {
        try
        {
            this->m_jointDensities.col(j) = this->m_distributions[j].probabilityDensityFunction(dataset) * this->m_spatialCoefficients->classCoefficients(j);
            m_scaleWeights.col(j) = this->m_distributions[j].scaleFunction(dataset);
        }
        catch (const std::exception &e)
        {
            std::stringstream s;
            s << "In function " << __PRETTY_FUNCTION__ << " ==> Component: " << j << std::endl << e.what();
            throw std::runtime_error(s.str());
        }
    }

#pragma omp parallel for
    for (int i = 0; i < this->m_observations; ++i)
    {
        const ArrayXd row = this->m_jointDensities.row(i);
        this->m_evidenceDensities(i) = row.sum();
        this->m_posteriorProbabilities.row(i) = row / this->m_evidenceDensities(i);
    }
    
    this->m_spatialCoefficients->updateDistributions();
}

template<int Dimensions>
void SpatiallyVariantFiniteMixtureModel<MultivariateTStudent, Dimensions>::updateParameters(const MatrixXd &dataset)
{
    for (int j = 0; j < this->m_components; ++j)
    {
        try
        {
            this->m_distributions[j].maximumLikelihoodEstimate(dataset, this->m_posteriorProbabilities.col(j), m_scaleWeights.col(j));
        }
        catch (const std::exception &e)
        {
            std::stringstream s;
            s << "In function " << __PRETTY_FUNCTION__ << " ==> Component: " << j << std::endl << e.what();
            throw std::runtime_error(s.str());
        }
    }

    this->m_spatialCoefficients->updateParameters(this->m_posteriorProbabilities);
}

/***************************** Template Specialization Gamma *****************************/

template<int Dimensions>
class SpatiallyVariantFiniteMixtureModel<Gamma, Dimensions> : public SpatiallyVariantFiniteMixtureModelBase<Gamma, Dimensions>
{
public:
    SpatiallyVariantFiniteMixtureModel();
    SpatiallyVariantFiniteMixtureModel(std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients);
    SpatiallyVariantFiniteMixtureModel(const std::vector<Gamma> &distributions, std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients);
    SpatiallyVariantFiniteMixtureModel(const SpatiallyVariantFiniteMixtureModel<Gamma, Dimensions> &other);
    
    SpatiallyVariantFiniteMixtureModel<Gamma, Dimensions> & operator = (const SpatiallyVariantFiniteMixtureModel<Gamma, Dimensions> &other);

    void initialization(const MatrixXd &dataset, const int K, const VectorXi &labels, const double regularization = 0);
    void updateDistributions(const MatrixXd &dataset);
    void updateParameters(const MatrixXd &dataset);
};

template<int Dimensions>
SpatiallyVariantFiniteMixtureModel<Gamma, Dimensions>::SpatiallyVariantFiniteMixtureModel()
: SpatiallyVariantFiniteMixtureModelBase<Gamma, Dimensions>()
{
}

template<int Dimensions>
SpatiallyVariantFiniteMixtureModel<Gamma, Dimensions>::SpatiallyVariantFiniteMixtureModel(std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients)
: SpatiallyVariantFiniteMixtureModelBase<Gamma, Dimensions>(std::move(spatialCoefficients))
{
}

template<int Dimensions>
SpatiallyVariantFiniteMixtureModel<Gamma, Dimensions>::SpatiallyVariantFiniteMixtureModel(const std::vector<Gamma> &distributions, std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients)
: SpatiallyVariantFiniteMixtureModelBase<Gamma, Dimensions>(distributions, std::move(spatialCoefficients))
{
}

template<int Dimensions>
SpatiallyVariantFiniteMixtureModel<Gamma, Dimensions>::SpatiallyVariantFiniteMixtureModel(const SpatiallyVariantFiniteMixtureModel<Gamma, Dimensions> &other)
: SpatiallyVariantFiniteMixtureModelBase<Gamma, Dimensions>(other)
{
}

template<int Dimensions>
SpatiallyVariantFiniteMixtureModel<Gamma, Dimensions>& SpatiallyVariantFiniteMixtureModel<Gamma, Dimensions>::operator = (const SpatiallyVariantFiniteMixtureModel<Gamma, Dimensions> &other)
{
    SpatiallyVariantFiniteMixtureModelBase<Gamma, Dimensions>::operator=(other);
    return *this;
}

template<int Dimensions>
void SpatiallyVariantFiniteMixtureModel<Gamma, Dimensions>::initialization(const MatrixXd &dataset, const int K, const VectorXi &labels, const double regularization)
{
    if (dataset.cols() != 1)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Invalid data dimensions for univariate Gamma Spatially Variant Finite Mixture Model" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    SpatiallyVariantFiniteMixtureModelBase<Gamma, Dimensions>::initialization(dataset, K, labels);
    
    for (int j = 0; j < this->m_components; ++j)
    {
        int n = 0;
        double temp1 = 0;
        double temp2 = 0;
        for (int i = 0; i < this->m_observations; ++i)
        {
            if (labels(i) == j)
            {
                temp1 += dataset(i, 0);
                temp2 += std::log(dataset(i, 0));
                ++n;
            }
        }
        const double s = std::log(temp1 / (double) n) - (temp2 / (double) n);
        const double k = (3.0 - s + std::sqrt(std::pow(s - 3.0, 2) + 24.0 * s)) / (12.0 * s);
        const double theta = temp1 / ((double) n * k);
        this->m_distributions.push_back(Gamma(k, theta));
    }
}

template<int Dimensions>
void SpatiallyVariantFiniteMixtureModel<Gamma, Dimensions>::updateDistributions(const MatrixXd &dataset)
{
    for (int j = 0; j < this->m_components; ++j)
    {
        try
        {
            this->m_jointDensities.col(j) = this->m_distributions[j].probabilityDensityFunction(dataset) * this->m_spatialCoefficients->classCoefficients(j);
        }
        catch (const std::exception &e)
        {
            std::stringstream s;
            s << "In function " << __PRETTY_FUNCTION__ << " ==> Component: " << j << std::endl << e.what();
            throw std::runtime_error(s.str());
        }
    }

#pragma omp parallel for
    for (int i = 0; i < this->m_observations; ++i)
    {
        const ArrayXd row = this->m_jointDensities.row(i);
        this->m_evidenceDensities(i) = row.sum();
        this->m_posteriorProbabilities.row(i) = row / this->m_evidenceDensities(i);
    }
    
    this->m_spatialCoefficients->updateDistributions();
}

template<int Dimensions>
void SpatiallyVariantFiniteMixtureModel<Gamma, Dimensions>::updateParameters(const MatrixXd &dataset)
{
    for (int j = 0; j < this->m_components; ++j)
    {
        try
        {
            this->m_distributions[j].maximumLikelihoodEstimate(dataset, this->m_posteriorProbabilities.col(j));
        }
        catch (const std::exception &e)
        {
            std::stringstream s;
            s << "In function " << __PRETTY_FUNCTION__ << " ==> Component: " << j << std::endl << e.what();
            throw std::runtime_error(s.str());
        }
    }

    this->m_spatialCoefficients->updateParameters(this->m_posteriorProbabilities);
}

#endif
