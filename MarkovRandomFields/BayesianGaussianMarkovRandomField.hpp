/*****************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                           */
/* Universidad Politecnica de Valencia, Spain                               */
/*                                                                          */
/* Copyright (C) 2020 Javier Juan Albarracin                                */
/*                                                                          */
/*****************************************************************************
* Bayesian Gaussian Markov Random Fields                                     *
*****************************************************************************/

#ifndef BAYESIANGAUSSIANMARKOVRANDOMFIELD_HPP
#define BAYESIANGAUSSIANMARKOVRANDOMFIELD_HPP

#define cimg_display 0
#include <CImg.h>
#include <omp.h>
#include "MarkovRandomField.hpp"
#include "GaussianMarkovRandomField.hpp"

using namespace Eigen;

/***************************** Bayesian Isotropic Gaussian Markov Random Field *****************************/

template<int Dimensions>
class BayesianIsotropicGaussianMarkovRandomField : public IsotropicGaussianMarkovRandomField<Dimensions>, public BayesianMarkovRandomField<Dimensions>
{
public:
    BayesianIsotropicGaussianMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    BayesianIsotropicGaussianMarkovRandomField(const BayesianIsotropicGaussianMarkovRandomField<Dimensions> &other);

    std::unique_ptr<MarkovRandomField<Dimensions>> clone() const;
    
    BayesianIsotropicGaussianMarkovRandomField<Dimensions> & operator = (const BayesianIsotropicGaussianMarkovRandomField<Dimensions> &other);
    
    void initialization(const ArrayXXd &coefficients);
    void updateDistributions();
    void updateParameters(const ArrayXXd &posteriorProbabilities);
    double logLikelihood();

private:
    BayesianIsotropicGaussianMarkovRandomField();
    
    void updateCoefficients(const ArrayXXd &posteriorProbabilities);
    void updateVariances();
};

/***************************** Implementation *****************************/

template<int Dimensions>
BayesianIsotropicGaussianMarkovRandomField<Dimensions>::BayesianIsotropicGaussianMarkovRandomField()
: IsotropicGaussianMarkovRandomField<Dimensions>(), BayesianMarkovRandomField<Dimensions>()
{
}

template<int Dimensions>
BayesianIsotropicGaussianMarkovRandomField<Dimensions>::BayesianIsotropicGaussianMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: IsotropicGaussianMarkovRandomField<Dimensions>(mask, classes, topology), BayesianMarkovRandomField<Dimensions>(this->m_coefficients)
{
}

template<int Dimensions>
BayesianIsotropicGaussianMarkovRandomField<Dimensions>::BayesianIsotropicGaussianMarkovRandomField(const BayesianIsotropicGaussianMarkovRandomField<Dimensions> &other)
: IsotropicGaussianMarkovRandomField<Dimensions>(other), BayesianMarkovRandomField<Dimensions>(other)
{
}

template<int Dimensions>
BayesianIsotropicGaussianMarkovRandomField<Dimensions>& BayesianIsotropicGaussianMarkovRandomField<Dimensions>::operator = (const BayesianIsotropicGaussianMarkovRandomField<Dimensions> &other)
{
    IsotropicGaussianMarkovRandomField<Dimensions>::operator=(other);
    BayesianMarkovRandomField<Dimensions>::operator=(other);
    
    return *this;
}

template<int Dimensions>
std::unique_ptr<MarkovRandomField<Dimensions>> BayesianIsotropicGaussianMarkovRandomField<Dimensions>::clone() const
{
    return std::unique_ptr<MarkovRandomField<Dimensions>>(new BayesianIsotropicGaussianMarkovRandomField<Dimensions>(*this));
}

template<int Dimensions>
void BayesianIsotropicGaussianMarkovRandomField<Dimensions>::initialization(const ArrayXXd &coefficients)
{
    if (coefficients.rows() != this->m_nodes && coefficients.cols() != this->m_classes)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent coefficients initialization" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    this->m_coefficients = coefficients;
    this->m_concentrations = coefficients;
    updateVariances();
}

template<int Dimensions>
void BayesianIsotropicGaussianMarkovRandomField<Dimensions>::updateDistributions()
{
}

template<int Dimensions>
void BayesianIsotropicGaussianMarkovRandomField<Dimensions>::updateParameters(const ArrayXXd &posteriorProbabilities)
{
    updateCoefficients(posteriorProbabilities);
    updateVariances();
}

template<int Dimensions>
void BayesianIsotropicGaussianMarkovRandomField<Dimensions>::updateCoefficients(const ArrayXXd &posteriorProbabilities)
{
    ArrayXXd newConcentrations(this->m_nodes, this->m_classes);
#pragma omp parallel for
    for (int i = 0; i < this->m_nodes; ++i)
    {
        const double Ai = this->m_concentrations.row(i).sum();
        for (int j = 0; j < this->m_classes; ++j)
        {
            const double Ai_j = Ai - this->m_concentrations(i, j);
            
            double ANi = 0;
            for (int n = 0; n < this->m_cliques; ++n)
                ANi += this->m_concentrations(this->neighbour(i, n), j);

            const double a = 1.0;
            const double b = Ai_j - (ANi / (double) this->m_cliques);
            const double c = -(Ai_j * ANi) / (double) this->m_cliques;
            const double d = -(posteriorProbabilities(i, j) * Ai_j * this->m_sigma(j)) / (2.0 * (double) this->m_cliques);
            
            double r1 = 0, r2 = 0, r3 = 0, i1 = 0, i2 = 0, i3 = 0, h = 0, alpha = 0;
            this->cubicSolver(a, b, c, d, r1, r2, r3, i1, i2, i3, h);

            if (h > 0)
                alpha = r1 < 0 ? SVFMM_ZERO : r1;
            else
            {
                alpha = std::max(std::max(r1, r2), r3);
                if (alpha < 0)
                    alpha = SVFMM_ZERO;
            }
            newConcentrations(i, j) = alpha;
        }
    }
#pragma omp parallel for
    for (int i = 0; i < this->m_nodes; i++)
    {
        this->m_concentrations.row(i) = newConcentrations.row(i);
        this->m_coefficients.row(i) = this->m_concentrations.row(i) / this->m_concentrations.row(i).sum();
    }
}

template<int Dimensions>
void BayesianIsotropicGaussianMarkovRandomField<Dimensions>::updateVariances()
{
#pragma omp parallel for
    for (int j = 0; j < this->m_classes; ++j)
    {
        double D = 0;
        for (int i = 0; i < this->m_nodes; ++i)
        {
            for (int n = 0; n < this->m_cliques; ++n)
            {
                const double diff = this->m_concentrations(i, j) - this->m_concentrations(this->neighbour(i, n), j);
                D += (diff * diff);
            }
        }
        this->m_sigma(j) = D / (double) (this->m_nodes * this->m_cliques);
    }
}

template<int Dimensions>
double BayesianIsotropicGaussianMarkovRandomField<Dimensions>::logLikelihood()
{
    const double NM = (double) (this->m_nodes * this->m_cliques);
    const double NKMLog2PIHalf = (double) (this->m_nodes * this->m_classes * this->m_cliques) * 0.918938533204673;
    
    ArrayXd logLikelihood(this->m_classes);
#pragma omp parallel for
    for (int j = 0; j < this->m_classes; ++j)
    {
        double D = 0;
        for (int i = 0; i < this->m_nodes; ++i)
        {
            for (int n = 0; n < this->m_cliques; ++n)
            {
                const double diff = this->m_concentrations(i, j) - this->m_concentrations(this->neighbour(i, n), j);
                D += (diff * diff);
            }
        }
        logLikelihood(j) = -(D / (2.0 * this->m_sigma(j))) - (NM * (std::log(this->m_sigma(j)) / 2.0));
    }
    return logLikelihood.sum() - NKMLog2PIHalf;
}

/***************************** Bayesian Anisotropic Gaussian Markov Random Field *****************************/

template<int Dimensions>
class BayesianAnisotropicGaussianMarkovRandomField : public AnisotropicGaussianMarkovRandomField<Dimensions>, public BayesianMarkovRandomField<Dimensions>
{
public:
    BayesianAnisotropicGaussianMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    BayesianAnisotropicGaussianMarkovRandomField(const BayesianAnisotropicGaussianMarkovRandomField<Dimensions> &other);

    std::unique_ptr<MarkovRandomField<Dimensions>> clone() const;
    
    BayesianAnisotropicGaussianMarkovRandomField<Dimensions> & operator = (const BayesianAnisotropicGaussianMarkovRandomField<Dimensions> &other);
    
    void initialization(const ArrayXXd &coefficients);
    void updateDistributions();
    void updateParameters(const ArrayXXd &posteriorProbabilities);
    double logLikelihood();

private:
    BayesianAnisotropicGaussianMarkovRandomField();
    
    void updateCoefficients(const ArrayXXd &posteriorProbabilities);
    void updateVariances();
};

/***************************** Implementation *****************************/

template<int Dimensions>
BayesianAnisotropicGaussianMarkovRandomField<Dimensions>::BayesianAnisotropicGaussianMarkovRandomField()
: AnisotropicGaussianMarkovRandomField<Dimensions>(), BayesianMarkovRandomField<Dimensions>()
{
}

template<int Dimensions>
BayesianAnisotropicGaussianMarkovRandomField<Dimensions>::BayesianAnisotropicGaussianMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: AnisotropicGaussianMarkovRandomField<Dimensions>(mask, classes, topology), BayesianMarkovRandomField<Dimensions>(this->m_coefficients)
{
}

template<int Dimensions>
BayesianAnisotropicGaussianMarkovRandomField<Dimensions>::BayesianAnisotropicGaussianMarkovRandomField(const BayesianAnisotropicGaussianMarkovRandomField<Dimensions> &other)
: AnisotropicGaussianMarkovRandomField<Dimensions>(other), BayesianMarkovRandomField<Dimensions>(other)
{
}

template<int Dimensions>
BayesianAnisotropicGaussianMarkovRandomField<Dimensions>& BayesianAnisotropicGaussianMarkovRandomField<Dimensions>::operator = (const BayesianAnisotropicGaussianMarkovRandomField<Dimensions> &other)
{
    AnisotropicGaussianMarkovRandomField<Dimensions>::operator=(other);
    BayesianMarkovRandomField<Dimensions>::operator=(other);
    
    return *this;
}

template<int Dimensions>
std::unique_ptr<MarkovRandomField<Dimensions>> BayesianAnisotropicGaussianMarkovRandomField<Dimensions>::clone() const
{
    return std::unique_ptr<MarkovRandomField<Dimensions>>(new BayesianAnisotropicGaussianMarkovRandomField<Dimensions>(*this));
}

template<int Dimensions>
void BayesianAnisotropicGaussianMarkovRandomField<Dimensions>::initialization(const ArrayXXd &coefficients)
{
    if (coefficients.rows() != this->m_nodes && coefficients.cols() != this->m_classes)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent coefficients initialization" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    this->m_coefficients = coefficients;
    this->m_concentrations = coefficients;
    updateVariances();
}

template<int Dimensions>
void BayesianAnisotropicGaussianMarkovRandomField<Dimensions>::updateDistributions()
{
}

template<int Dimensions>
void BayesianAnisotropicGaussianMarkovRandomField<Dimensions>::updateParameters(const ArrayXXd &posteriorProbabilities)
{
    updateCoefficients(posteriorProbabilities);
    updateVariances();
}

template<int Dimensions>
void BayesianAnisotropicGaussianMarkovRandomField<Dimensions>::updateCoefficients(const ArrayXXd &posteriorProbabilities)
{
    ArrayXXd newConcentrations(this->m_nodes, this->m_classes);
#pragma omp parallel for
    for (int i = 0; i < this->m_nodes; ++i)
    {
        const double Ai = this->m_concentrations.row(i).sum();
        for (int j = 0; j < this->m_classes; ++j)
        {
            const double Ai_j = Ai - this->m_concentrations(i, j);
            
            double weightsNid = 0;
            double ANid = 0;
            for (int d = 0; d < this->m_directions; ++d)
            {
                double ANi = 0;
                for (int n = 0; n < this->m_cliques; ++n)
                    ANi += this->m_concentrations(this->neighbour(i, d, n), j);

                ANid += (ANi / this->m_sigma(j, d));
                weightsNid += ((double) this->m_cliques / this->m_sigma(j, d));
            }

            const double a = 1.0;
            const double b = Ai_j - (ANid / weightsNid);
            const double c = -(Ai_j * ANid) / weightsNid;
            const double e = -((posteriorProbabilities(i, j) * Ai_j) / (2.0 * weightsNid));
            
            double r1 = 0, r2 = 0, r3 = 0, i1 = 0, i2 = 0, i3 = 0, h = 0, alpha = 0;
            this->cubicSolver(a, b, c, e, r1, r2, r3, i1, i2, i3, h);
            
            if (h > 0)
                alpha = r1 < 0 ? SVFMM_ZERO : r1;
            else
            {
                alpha = std::max(std::max(r1, r2), r3);
                if (alpha < 0)
                    alpha = SVFMM_ZERO;
            }
            newConcentrations(i, j) = alpha;
        }
    }
#pragma omp parallel for
    for (int i = 0; i < this->m_nodes; i++)
    {
        this->m_concentrations.row(i) = newConcentrations.row(i);
        this->m_coefficients.row(i) = this->m_concentrations.row(i) / this->m_concentrations.row(i).sum();
    }
}

template<int Dimensions>
void BayesianAnisotropicGaussianMarkovRandomField<Dimensions>::updateVariances()
{
#pragma omp parallel for
    for (int j = 0; j < this->m_classes; ++j)
    {
        for (int d = 0; d < this->m_directions; ++d)
        {
            double D = 0;
            for (int i = 0; i < this->m_nodes; ++i)
            {
                for (int n = 0; n < this->m_cliques; ++n)
                {
                    const double diff = this->m_concentrations(i, j) - this->m_concentrations(this->neighbour(i, d, n), j);
                    D += (diff * diff);
                }
            }
            this->m_sigma(j, d) = D / (double) (this->m_nodes * this->m_cliques);
        }
    }
}

template<int Dimensions>
double BayesianAnisotropicGaussianMarkovRandomField<Dimensions>::logLikelihood()
{
    const double NM = (double) (this->m_nodes * this->m_cliques);
    const double NKDMLog2PIHalf = (double) (this->m_nodes * this->m_classes * this->m_directions * this->m_cliques) * 0.918938533204673;
    
    ArrayXd logLikelihood(this->m_classes);
#pragma omp parallel for
    for (int j = 0; j < this->m_classes; ++j)
    {
        double sumLogLikelihoodPerDirection = 0;
        for (int d = 0; d < this->m_directions; ++d)
        {
            double D = 0;
            for (int i = 0; i < this->m_nodes; ++i)
            {
                for (int n = 0; n < this->m_cliques; ++n)
                {
                    const double diff = this->m_concentrations(i, j) - this->m_concentrations(this->neighbour(i, d, n), j);
                    D += (diff * diff);
                }
            }
            sumLogLikelihoodPerDirection += -(D / (2.0 * this->m_sigma(j, d))) - (NM * (std::log(this->m_sigma(j, d)) / 2.0));
        }
        logLikelihood(j) = sumLogLikelihoodPerDirection;
    }
    return logLikelihood.sum() - NKDMLog2PIHalf;
}

#endif