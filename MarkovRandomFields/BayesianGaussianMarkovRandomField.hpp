/*****************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                           */
/* Universidad Politecnica de Valencia, Spain                               */
/*                                                                          */
/* Copyright (C) 2020 Javier Juan Albarracin                                */
/*                                                                          */
/*****************************************************************************
* Bayesian Gauss Markov Random Fields                                        *
*****************************************************************************/

#ifndef BAYESIANGAUSSMARKOVRANDOMFIELD_HPP
#define BAYESIANGAUSSMARKOVRANDOMFIELD_HPP

#define cimg_display 0
#include <CImg.h>
#include <omp.h>
#include "MarkovRandomField.hpp"
#include "GaussMarkovRandomField.hpp"

using namespace Eigen;

/***************************** Bayesian Isotropic Gauss Markov Random Field *****************************/

template<int Dimensions>
class BayesianIsotropicGaussMarkovRandomField : public IsotropicGaussMarkovRandomField<Dimensions>, public BayesianMarkovRandomField<Dimensions>
{
public:
    BayesianIsotropicGaussMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    BayesianIsotropicGaussMarkovRandomField(const BayesianIsotropicGaussMarkovRandomField<Dimensions> &other);

    std::unique_ptr<MarkovRandomField<Dimensions>> clone() const;
    
    BayesianIsotropicGaussMarkovRandomField<Dimensions> & operator = (const BayesianIsotropicGaussMarkovRandomField<Dimensions> &other);
    
    void initialization(const ArrayXXd &coefficients);
    void updateDistributions();
    void updateParameters(const ArrayXXd &posteriorProbabilities);
    double logLikelihood();

private:
    BayesianIsotropicGaussMarkovRandomField();
    
    void updateCoefficients(const ArrayXXd &posteriorProbabilities);
    void updateVariances();
};

/***************************** Implementation *****************************/

template<int Dimensions>
BayesianIsotropicGaussMarkovRandomField<Dimensions>::BayesianIsotropicGaussMarkovRandomField()
: IsotropicGaussMarkovRandomField<Dimensions>(), BayesianMarkovRandomField<Dimensions>()
{
}

template<int Dimensions>
BayesianIsotropicGaussMarkovRandomField<Dimensions>::BayesianIsotropicGaussMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: IsotropicGaussMarkovRandomField<Dimensions>(mask, classes, topology), BayesianMarkovRandomField<Dimensions>(this->m_coefficients)
{
}

template<int Dimensions>
BayesianIsotropicGaussMarkovRandomField<Dimensions>::BayesianIsotropicGaussMarkovRandomField(const BayesianIsotropicGaussMarkovRandomField<Dimensions> &other)
: IsotropicGaussMarkovRandomField<Dimensions>(other), BayesianMarkovRandomField<Dimensions>(other)
{
}

template<int Dimensions>
BayesianIsotropicGaussMarkovRandomField<Dimensions>& BayesianIsotropicGaussMarkovRandomField<Dimensions>::operator = (const BayesianIsotropicGaussMarkovRandomField<Dimensions> &other)
{
    IsotropicGaussMarkovRandomField<Dimensions>::operator=(other);
    BayesianMarkovRandomField<Dimensions>::operator=(other);
    
    return *this;
}

template<int Dimensions>
std::unique_ptr<MarkovRandomField<Dimensions>> BayesianIsotropicGaussMarkovRandomField<Dimensions>::clone() const
{
    return std::unique_ptr<MarkovRandomField<Dimensions>>(new BayesianIsotropicGaussMarkovRandomField<Dimensions>(*this));
}

template<int Dimensions>
void BayesianIsotropicGaussMarkovRandomField<Dimensions>::initialization(const ArrayXXd &coefficients)
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
void BayesianIsotropicGaussMarkovRandomField<Dimensions>::updateDistributions()
{
}

template<int Dimensions>
void BayesianIsotropicGaussMarkovRandomField<Dimensions>::updateParameters(const ArrayXXd &posteriorProbabilities)
{
    updateCoefficients(posteriorProbabilities);
    updateVariances();
}

template<int Dimensions>
void BayesianIsotropicGaussMarkovRandomField<Dimensions>::updateCoefficients(const ArrayXXd &posteriorProbabilities)
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
void BayesianIsotropicGaussMarkovRandomField<Dimensions>::updateVariances()
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
double BayesianIsotropicGaussMarkovRandomField<Dimensions>::logLikelihood()
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

/***************************** Bayesian Anisotropic Gauss Markov Random Field *****************************/

template<int Dimensions>
class BayesianAnisotropicGaussMarkovRandomField : public AnisotropicGaussMarkovRandomField<Dimensions>, public BayesianMarkovRandomField<Dimensions>
{
public:
    BayesianAnisotropicGaussMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    BayesianAnisotropicGaussMarkovRandomField(const BayesianAnisotropicGaussMarkovRandomField<Dimensions> &other);

    std::unique_ptr<MarkovRandomField<Dimensions>> clone() const;
    
    BayesianAnisotropicGaussMarkovRandomField<Dimensions> & operator = (const BayesianAnisotropicGaussMarkovRandomField<Dimensions> &other);
    
    void initialization(const ArrayXXd &coefficients);
    void updateDistributions();
    void updateParameters(const ArrayXXd &posteriorProbabilities);
    double logLikelihood();

private:
    BayesianAnisotropicGaussMarkovRandomField();
    
    void updateCoefficients(const ArrayXXd &posteriorProbabilities);
    void updateVariances();
};

/***************************** Implementation *****************************/

template<int Dimensions>
BayesianAnisotropicGaussMarkovRandomField<Dimensions>::BayesianAnisotropicGaussMarkovRandomField()
: AnisotropicGaussMarkovRandomField<Dimensions>(), BayesianMarkovRandomField<Dimensions>()
{
}

template<int Dimensions>
BayesianAnisotropicGaussMarkovRandomField<Dimensions>::BayesianAnisotropicGaussMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: AnisotropicGaussMarkovRandomField<Dimensions>(mask, classes, topology), BayesianMarkovRandomField<Dimensions>(this->m_coefficients)
{
}

template<int Dimensions>
BayesianAnisotropicGaussMarkovRandomField<Dimensions>::BayesianAnisotropicGaussMarkovRandomField(const BayesianAnisotropicGaussMarkovRandomField<Dimensions> &other)
: AnisotropicGaussMarkovRandomField<Dimensions>(other), BayesianMarkovRandomField<Dimensions>(other)
{
}

template<int Dimensions>
BayesianAnisotropicGaussMarkovRandomField<Dimensions>& BayesianAnisotropicGaussMarkovRandomField<Dimensions>::operator = (const BayesianAnisotropicGaussMarkovRandomField<Dimensions> &other)
{
    AnisotropicGaussMarkovRandomField<Dimensions>::operator=(other);
    BayesianMarkovRandomField<Dimensions>::operator=(other);
    
    return *this;
}

template<int Dimensions>
std::unique_ptr<MarkovRandomField<Dimensions>> BayesianAnisotropicGaussMarkovRandomField<Dimensions>::clone() const
{
    return std::unique_ptr<MarkovRandomField<Dimensions>>(new BayesianAnisotropicGaussMarkovRandomField<Dimensions>(*this));
}

template<int Dimensions>
void BayesianAnisotropicGaussMarkovRandomField<Dimensions>::initialization(const ArrayXXd &coefficients)
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
void BayesianAnisotropicGaussMarkovRandomField<Dimensions>::updateDistributions()
{
}

template<int Dimensions>
void BayesianAnisotropicGaussMarkovRandomField<Dimensions>::updateParameters(const ArrayXXd &posteriorProbabilities)
{
    updateCoefficients(posteriorProbabilities);
    updateVariances();
}

template<int Dimensions>
void BayesianAnisotropicGaussMarkovRandomField<Dimensions>::updateCoefficients(const ArrayXXd &posteriorProbabilities)
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
void BayesianAnisotropicGaussMarkovRandomField<Dimensions>::updateVariances()
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
double BayesianAnisotropicGaussMarkovRandomField<Dimensions>::logLikelihood()
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