/*****************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                           */
/* Universidad Politecnica de Valencia, Spain                               */
/*                                                                          */
/* Copyright (C) 2020 Javier Juan Albarracin                                */
/*                                                                          */
/*****************************************************************************
* Frequentist Gauss Markov Random Fields                                     *
*****************************************************************************/

#ifndef FREQUENTISTGAUSSMARKOVRANDOMFIELD_HPP
#define FREQUENTISTGAUSSMARKOVRANDOMFIELD_HPP

#define cimg_display 0
#include <CImg.h>
#include <omp.h>
#include "MarkovRandomField.hpp"
#include "GaussMarkovRandomField.hpp"

using namespace Eigen;

/***************************** Frequentist Isotropic Gauss Markov Random Field *****************************/

template<int Dimensions>
class FrequentistIsotropicGaussMarkovRandomField : public IsotropicGaussMarkovRandomField<Dimensions>, public FrequentistMarkovRandomField<Dimensions>
{
public:
    FrequentistIsotropicGaussMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    FrequentistIsotropicGaussMarkovRandomField(const FrequentistIsotropicGaussMarkovRandomField<Dimensions> &other);

    std::unique_ptr<MarkovRandomField<Dimensions>> clone() const;
    
    FrequentistIsotropicGaussMarkovRandomField<Dimensions> & operator = (const FrequentistIsotropicGaussMarkovRandomField<Dimensions> &other);
    
    void initialization(const ArrayXXd &coefficients);
    void updateDistributions();
    void updateParameters(const ArrayXXd &posteriorProbabilities);
    double logLikelihood();

private:
    FrequentistIsotropicGaussMarkovRandomField();
    
    void updateCoefficients(const ArrayXXd &posteriorProbabilities);
    void updateVariances();
};

/***************************** Implementation *****************************/

template<int Dimensions>
FrequentistIsotropicGaussMarkovRandomField<Dimensions>::FrequentistIsotropicGaussMarkovRandomField()
: IsotropicGaussMarkovRandomField<Dimensions>(), FrequentistMarkovRandomField<Dimensions>()
{
}

template<int Dimensions>
FrequentistIsotropicGaussMarkovRandomField<Dimensions>::FrequentistIsotropicGaussMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: IsotropicGaussMarkovRandomField<Dimensions>(mask, classes, topology), FrequentistMarkovRandomField<Dimensions>()
{
}

template<int Dimensions>
FrequentistIsotropicGaussMarkovRandomField<Dimensions>::FrequentistIsotropicGaussMarkovRandomField(const FrequentistIsotropicGaussMarkovRandomField<Dimensions> &other)
: IsotropicGaussMarkovRandomField<Dimensions>(other), FrequentistMarkovRandomField<Dimensions>(other)
{
}

template<int Dimensions>
FrequentistIsotropicGaussMarkovRandomField<Dimensions>& FrequentistIsotropicGaussMarkovRandomField<Dimensions>::operator = (const FrequentistIsotropicGaussMarkovRandomField<Dimensions> &other)
{
    IsotropicGaussMarkovRandomField<Dimensions>::operator=(other);
    FrequentistMarkovRandomField<Dimensions>::operator=(other);
    
    return *this;
}

template<int Dimensions>
std::unique_ptr<MarkovRandomField<Dimensions>> FrequentistIsotropicGaussMarkovRandomField<Dimensions>::clone() const
{
    return std::unique_ptr<MarkovRandomField<Dimensions>>(new FrequentistIsotropicGaussMarkovRandomField<Dimensions>(*this));
}

template<int Dimensions>
void FrequentistIsotropicGaussMarkovRandomField<Dimensions>::initialization(const ArrayXXd &coefficients)
{
    if (coefficients.rows() != this->m_nodes && coefficients.cols() != this->m_classes)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent coefficients initialization" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    this->m_coefficients = coefficients;
    updateVariances();
}

template<int Dimensions>
void FrequentistIsotropicGaussMarkovRandomField<Dimensions>::updateDistributions()
{
}

template<int Dimensions>
void FrequentistIsotropicGaussMarkovRandomField<Dimensions>::updateParameters(const ArrayXXd &posteriorProbabilities)
{
    updateCoefficients(posteriorProbabilities);
    updateVariances();
}

template<int Dimensions>
void FrequentistIsotropicGaussMarkovRandomField<Dimensions>::updateCoefficients(const ArrayXXd &posteriorProbabilities)
{
    const ArrayXd weights = ArrayXd::Ones(this->m_classes);
    ArrayXXd newCoefficients(this->m_nodes, this->m_classes);
#pragma omp parallel for
    for (int i = 0; i < this->m_nodes; ++i)
    {
        ArrayXd coeffs_i(this->m_classes);
        for (int j = 0; j < this->m_classes; ++j)
        {
            double coeffsNi = 0;
            for (int n = 0; n < this->m_cliques; ++n)
                coeffsNi += this->m_coefficients(this->neighbour(i, n), j);

            const double a = 1.0;
            const double b = -(coeffsNi / (double) this->m_cliques);
            const double c = -(posteriorProbabilities(i, j) * this->m_sigma(j)) / (2.0 * (double) this->m_cliques);
            double r1, r2;

            if (this->cuadraticSolver(a, b, c, r1, r2))
                coeffs_i(j) = r1 > r2 ? r1 : r2;
            else
                coeffs_i(j) = SVFMM_ZERO;
        }
        newCoefficients.row(i) = this->conjugateProjection(coeffs_i, weights);
    }
    this->m_coefficients = newCoefficients;
}

template<int Dimensions>
void FrequentistIsotropicGaussMarkovRandomField<Dimensions>::updateVariances()
{
#pragma omp parallel for
    for (int j = 0; j < this->m_classes; ++j)
    {
        double D = 0;
        for (int i = 0; i < this->m_nodes; ++i)
        {
            for (int n = 0; n < this->m_cliques; ++n)
            {
                const double diff = this->m_coefficients(i, j) - this->m_coefficients(this->neighbour(i, n), j);
                D += (diff * diff);
            }
        }
        this->m_sigma(j) = D / (double) (this->m_nodes * this->m_cliques);
    }
}

template<int Dimensions>
double FrequentistIsotropicGaussMarkovRandomField<Dimensions>::logLikelihood()
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
                const double diff = this->m_coefficients(i, j) - this->m_coefficients(this->neighbour(i, n), j);
                D += (diff * diff);
            }
        }
        logLikelihood(j) = -(D / (2.0 * this->m_sigma(j))) - (NM * (std::log(this->m_sigma(j)) / 2.0));
    }
    return logLikelihood.sum() - NKMLog2PIHalf;
}

/***************************** Frequentist Anisotropic Gaussian Markov Random Field *****************************/

template<int Dimensions>
class FrequentistAnisotropicGaussMarkovRandomField : public AnisotropicGaussMarkovRandomField<Dimensions>, public FrequentistMarkovRandomField<Dimensions>
{
public:
    FrequentistAnisotropicGaussMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    FrequentistAnisotropicGaussMarkovRandomField(const FrequentistAnisotropicGaussMarkovRandomField<Dimensions> &other);

    std::unique_ptr<MarkovRandomField<Dimensions>> clone() const;
    
    FrequentistAnisotropicGaussMarkovRandomField<Dimensions> & operator = (const FrequentistAnisotropicGaussMarkovRandomField<Dimensions> &other);
    
    void initialization(const ArrayXXd &coefficients);
    void updateDistributions();
    void updateParameters(const ArrayXXd &posteriorProbabilities);
    double logLikelihood();

private:
    FrequentistAnisotropicGaussMarkovRandomField();
    
    void updateCoefficients(const ArrayXXd &posteriorProbabilities);
    void updateVariances();
};

/***************************** Implementation *****************************/

template<int Dimensions>
FrequentistAnisotropicGaussMarkovRandomField<Dimensions>::FrequentistAnisotropicGaussMarkovRandomField()
: AnisotropicGaussMarkovRandomField<Dimensions>(), FrequentistMarkovRandomField<Dimensions>()
{
}

template<int Dimensions>
FrequentistAnisotropicGaussMarkovRandomField<Dimensions>::FrequentistAnisotropicGaussMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: AnisotropicGaussMarkovRandomField<Dimensions>(mask, classes, topology), FrequentistMarkovRandomField<Dimensions>()
{
}

template<int Dimensions>
FrequentistAnisotropicGaussMarkovRandomField<Dimensions>::FrequentistAnisotropicGaussMarkovRandomField(const FrequentistAnisotropicGaussMarkovRandomField<Dimensions> &other)
: AnisotropicGaussMarkovRandomField<Dimensions>(other), FrequentistMarkovRandomField<Dimensions>(other)
{
}

template<int Dimensions>
FrequentistAnisotropicGaussMarkovRandomField<Dimensions>& FrequentistAnisotropicGaussMarkovRandomField<Dimensions>::operator = (const FrequentistAnisotropicGaussMarkovRandomField<Dimensions> &other)
{
    AnisotropicGaussMarkovRandomField<Dimensions>::operator=(other);
    FrequentistMarkovRandomField<Dimensions>::operator=(other);
    
    return *this;
}

template<int Dimensions>
std::unique_ptr<MarkovRandomField<Dimensions>> FrequentistAnisotropicGaussMarkovRandomField<Dimensions>::clone() const
{
    return std::unique_ptr<MarkovRandomField<Dimensions>>(new FrequentistAnisotropicGaussMarkovRandomField<Dimensions>(*this));
}

template<int Dimensions>
void FrequentistAnisotropicGaussMarkovRandomField<Dimensions>::initialization(const ArrayXXd &coefficients)
{
    if (coefficients.rows() != this->m_nodes && coefficients.cols() != this->m_classes)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent coefficients initialization" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    this->m_coefficients = coefficients;
    updateVariances();
}

template<int Dimensions>
void FrequentistAnisotropicGaussMarkovRandomField<Dimensions>::updateDistributions()
{
}

template<int Dimensions>
void FrequentistAnisotropicGaussMarkovRandomField<Dimensions>::updateParameters(const ArrayXXd &posteriorProbabilities)
{
    updateCoefficients(posteriorProbabilities);
    updateVariances();
}

template<int Dimensions>
void FrequentistAnisotropicGaussMarkovRandomField<Dimensions>::updateCoefficients(const ArrayXXd &posteriorProbabilities)
{
    const ArrayXd weights = ArrayXd::Ones(this->m_classes);
    ArrayXXd newCoefficients(this->m_nodes, this->m_classes);
#pragma omp parallel for
    for (int i = 0; i < this->m_nodes; ++i)
    {
        ArrayXd coeffs_i(this->m_classes);
        for (int j = 0; j < this->m_classes; ++j)
        {
            double weightsNid = 0;
            double coeffsNid = 0;
            for (int d = 0; d < this->m_directions; ++d)
            {
                double coeffsNi = 0;
                for (int n = 0; n < this->m_cliques; ++n)
                    coeffsNi += this->m_coefficients(this->neighbour(i, d, n), j);

                coeffsNid += (coeffsNi / this->m_sigma(j, d));
                weightsNid += ((double) this->m_cliques / this->m_sigma(j, d));
            }

            const double a = 1.0;
            const double b = -(coeffsNid / weightsNid);
            const double c = -(posteriorProbabilities(i, j) / (2.0 * weightsNid));
            double r1, r2;

            if (this->cuadraticSolver(a, b, c, r1, r2))
                coeffs_i(j) = r1 > r2 ? r1 : r2;
            else
                coeffs_i(j) = SVFMM_ZERO;
        }
        newCoefficients.row(i) = this->conjugateProjection(coeffs_i, weights);
    }
    this->m_coefficients = newCoefficients;
}

template<int Dimensions>
void FrequentistAnisotropicGaussMarkovRandomField<Dimensions>::updateVariances()
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
                    const double diff = this->m_coefficients(i, j) - this->m_coefficients(this->neighbour(i, d, n), j);
                    D += (diff * diff);
                }
            }
            this->m_sigma(j, d) = D / (double) (this->m_nodes * this->m_cliques);
        }
    }
}

template<int Dimensions>
double FrequentistAnisotropicGaussMarkovRandomField<Dimensions>::logLikelihood()
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
                    const double diff = this->m_coefficients(i, j) - this->m_coefficients(this->neighbour(i, d, n), j);
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