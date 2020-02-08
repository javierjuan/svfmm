/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Frequentist Non Local Markov Random Fields                               *
***************************************************************************/

#ifndef FREQUENTISTNONLOCALMARKOVRANDOMFIELD_HPP
#define FREQUENTISTNONLOCALMARKOVRANDOMFIELD_HPP

#define cimg_display 0
#include <CImg.h>
#include <omp.h>
#include "MarkovRandomField.hpp"
#include "NonLocalMarkovRandomField.hpp"
#include "../Constants.hpp"

using namespace Eigen;

/***************************** Frequentist Isotropic Non Local Markov Random Field *****************************/

template<int Dimensions, bool PatchBased = false>
class FrequentistIsotropicNonLocalMarkovRandomField : public IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>, public FrequentistMarkovRandomField<Dimensions>
{
public:
    FrequentistIsotropicNonLocalMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    FrequentistIsotropicNonLocalMarkovRandomField(const FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other);
    
    std::unique_ptr<MarkovRandomField<Dimensions>> clone() const;
    
    FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased> & operator = (const FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other);
    
    void initialization(const ArrayXXd &coefficients);
    void updateDistributions();
    void updateParameters(const ArrayXXd &posteriorProbabilities);
    double logLikelihood();

private:
    FrequentistIsotropicNonLocalMarkovRandomField();
    
    void updateWeights();
    void updateCoefficients(const ArrayXXd &posteriorProbabilities);
    void updateVariances();
    
    inline double stdQuantitativeChi2Test(const int index1, const int index2, const int component) const;
    inline double stdQuantitativeChi2TestPatch(const int index1, const int index2, const int component) const;
};

/***************************** Implementation *****************************/

template<int Dimensions, bool PatchBased>
FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::FrequentistIsotropicNonLocalMarkovRandomField()
: IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>(), FrequentistMarkovRandomField<Dimensions>()
{
}

template<int Dimensions, bool PatchBased>
FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::FrequentistIsotropicNonLocalMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>(mask, classes, topology), FrequentistMarkovRandomField<Dimensions>()
{
}

template<int Dimensions, bool PatchBased>
FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::FrequentistIsotropicNonLocalMarkovRandomField(const FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other)
: IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>(other), FrequentistMarkovRandomField<Dimensions>(other)
{
}

template<int Dimensions, bool PatchBased>
FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>& FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::operator = (const FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other)
{
    IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::operator=(other);
    FrequentistMarkovRandomField<Dimensions>::operator=(other);
    
    return *this;
}

template<int Dimensions, bool PatchBased>
std::unique_ptr<MarkovRandomField<Dimensions>> FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::clone() const
{
    return std::unique_ptr<MarkovRandomField<Dimensions>>(new FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>(*this));
}

template<int Dimensions, bool PatchBased>
void FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::initialization(const ArrayXXd &coefficients)
{
    if (coefficients.rows() != this->m_nodes && coefficients.cols() != this->m_classes)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent coefficients initialization" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    this->m_coefficients = coefficients;
    this->m_weights.setConstant(this->chi2pdf(this->m_nu));
    updateVariances();
    updateWeights();
}

template<int Dimensions, bool PatchBased>
void FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateDistributions()
{
    updateWeights();
}

template<int Dimensions, bool PatchBased>
void FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateParameters(const ArrayXXd &posteriorProbabilities)
{
    updateCoefficients(posteriorProbabilities);
    updateVariances();
}

template<int Dimensions, bool PatchBased>
double FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::stdQuantitativeChi2Test(const int index1, const int index2, const int component) const
{
    if (index1 == index2)
        return 0;
    
    const double diff = this->m_coefficients(index1, component) - this->m_coefficients(index2, component);
    return (diff * diff) / (double)(2.0 * this->m_chi2Sigma(component));
}

template<int Dimensions, bool PatchBased>
double FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::stdQuantitativeChi2TestPatch(const int index1, const int index2, const int component) const
{
    if (index1 == index2)
        return 0;
    
    double z = 0;
    for (int n = 0; n < this->m_patchSize; ++n)
    {
        const double diff = this->m_coefficients(this->m_patches(index1, n), component) - this->m_coefficients(this->m_patches(index2, n), component);
        z += (diff * diff);
    }
    return z / (double) (2.0 * this->m_chi2Sigma(component) * this->m_patchSize);
}

template<int Dimensions, bool PatchBased>
void FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateWeights()
{
#pragma omp parallel for
    for (int i = 0; i < this->m_nodes; ++i)
    {
        for (int j = 0; j < this->m_classes; ++j)
        {
            for (int n = 0; n < this->m_cliques; ++n)
            {
                double z = 0;
                if (PatchBased)
                    z = stdQuantitativeChi2TestPatch(i, this->neighbour(i, n), j);
                else
                    z = stdQuantitativeChi2Test(i, this->neighbour(i, n), j);
                z = z < SVFMM_CHI2_DEFAULT_MIN_Z ? SVFMM_CHI2_DEFAULT_MIN_Z : z;
                z = z > SVFMM_CHI2_DEFAULT_MAX_Z ? SVFMM_CHI2_DEFAULT_MAX_Z : z;
                this->m_weights(i, n, j) = (float) this->chi2pdf(z);
            }
        }
    }
}

template<int Dimensions, bool PatchBased>
void FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateCoefficients(const ArrayXXd &posteriorProbabilities)
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
            double weightsNi = 0;
            for (int n = 0; n < this->m_cliques; ++n)
            {
                coeffsNi += (this->m_coefficients(this->neighbour(i, n), j) * (double) this->m_weights(i, n, j));
                weightsNi += (double) this->m_weights(i, n, j);
            }

            const double a = 1.0;
            const double b = -(coeffsNi / weightsNi);
            const double c = -(posteriorProbabilities(i, j) * this->m_sigma(j)) / (2.0 * weightsNi);
            double r1, r2;

            if (this->cuadraticSolver(a, b, c, r1, r2))
                coeffs_i(j) = r1;
            else
                coeffs_i(j) = SVFMM_ZERO;
        }
        newCoefficients.row(i) = this->conjugateProjection(coeffs_i, weights);
    }
    this->m_coefficients = newCoefficients;
}

template<int Dimensions, bool PatchBased>
void FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateVariances()
{
#pragma omp parallel for
    for (int j = 0; j < this->m_classes; ++j)
    {
        double DNormal = 0;
        double DChi2 = 0;
        for (int i = 0; i < this->m_nodes; ++i)
        {
            for (int n = 0; n < this->m_cliques; ++n)
            {
                const double diff = this->m_coefficients(i, j) - this->m_coefficients(this->neighbour(i, n), j);
                DNormal += ((diff * diff) * (double) this->m_weights(i, n, j));
            }
            
            for (int n = 0; n < this->m_patchSize; ++n)
            {
                const int ni = this->m_patches(i, n);
                if (PatchBased)
                {
                    double patchDiff = 0;
                    for (int p = 0; p < this->m_patchSize; ++p)
                    {
                        const double diff = this->m_coefficients(this->m_patches(i, p), j) - this->m_coefficients(this->m_patches(ni, p), j);
                        patchDiff = (diff * diff);
                    }
                    DChi2 += (patchDiff / (double) this->m_patchSize);
                }
                else
                {
                    const double diff = this->m_coefficients(i, j) - this->m_coefficients(ni, j);
                    DChi2 += (diff * diff);
                }
            }
        }
        this->m_sigma(j) = DNormal / (double) (this->m_nodes * this->m_cliques);
        this->m_chi2Sigma(j) = DChi2 / (double) (this->m_nodes * this->m_patchSize);
    }
}

template<int Dimensions, bool PatchBased>
double FrequentistIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::logLikelihood()
{
    const double NM = (double) (this->m_nodes * this->m_cliques);
    const double NKM = (double) (this->m_nodes * this->m_classes * this->m_cliques);
    const double NKMLog2PIHalf = NKM * 0.918938533204673;
    
    ArrayXd logLikelihood(this->m_classes);
#pragma omp parallel for
    for (int j = 0; j < this->m_classes; ++j)
    {
        double D = 0;
        double sumLogU = 0;
        for (int i = 0; i < this->m_nodes; ++i)
        {
            for (int n = 0; n < this->m_cliques; ++n)
            {
                const double u = (double) this->m_weights(i, n, j);
                const double diff = this->m_coefficients(i, j) - this->m_coefficients(this->neighbour(i, n), j);
                D += (diff * diff * u);
                sumLogU += std::log(u);
            }
        }
        logLikelihood(j) = -(D / (2.0 * this->m_sigma(j))) + (sumLogU / 2.0) - (NM * (std::log(this->m_sigma(j)) / 2.0));
    }
    return logLikelihood.sum() - NKMLog2PIHalf;
}

/***************************** Frequentist Anisotropic Non Local Markov Random Field *****************************/

template<int Dimensions, bool PatchBased = false>
class FrequentistAnisotropicNonLocalMarkovRandomField : public AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>, public FrequentistMarkovRandomField<Dimensions>
{
public:
    FrequentistAnisotropicNonLocalMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    FrequentistAnisotropicNonLocalMarkovRandomField(const FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other);

    std::unique_ptr<MarkovRandomField<Dimensions>> clone() const;
    
    FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased> & operator = (const FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other);
    
    void initialization(const ArrayXXd &coefficients);
    void updateDistributions();
    void updateParameters(const ArrayXXd &posteriorProbabilities);
    double logLikelihood();

private:
    FrequentistAnisotropicNonLocalMarkovRandomField();
    
    void updateWeights();
    void updateCoefficients(const ArrayXXd &posteriorProbabilities);
    void updateVariances();
    
    inline double stdQuantitativeChi2Test(const int index1, const int index2, const int component) const;
    inline double stdQuantitativeChi2TestPatch(const int index1, const int index2, const int component) const;
};

/***************************** Implementation *****************************/

template<int Dimensions, bool PatchBased>
FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::FrequentistAnisotropicNonLocalMarkovRandomField()
: AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>(), FrequentistMarkovRandomField<Dimensions>()
{
}

template<int Dimensions, bool PatchBased>
FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::FrequentistAnisotropicNonLocalMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>(mask, classes, topology), FrequentistMarkovRandomField<Dimensions>()
{
}

template<int Dimensions, bool PatchBased>
FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::FrequentistAnisotropicNonLocalMarkovRandomField(const FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other)
: AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>(other), FrequentistMarkovRandomField<Dimensions>(other)
{
}

template<int Dimensions, bool PatchBased>
FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>& FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::operator = (const FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other)
{
    AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::operator=(other);
    FrequentistMarkovRandomField<Dimensions>::operator=(other);
    
    return *this;
}

template<int Dimensions, bool PatchBased>
std::unique_ptr<MarkovRandomField<Dimensions>> FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::clone() const
{
    return std::unique_ptr<MarkovRandomField<Dimensions>>(new FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>(*this));
}

template<int Dimensions, bool PatchBased>
void FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::initialization(const ArrayXXd &coefficients)
{
    if (coefficients.rows() != this->m_nodes && coefficients.cols() != this->m_classes)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent coefficients initialization" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    this->m_coefficients = coefficients;
    this->m_weights.setConstant(this->chi2pdf(this->m_nu));
    updateVariances();
    updateWeights();
}

template<int Dimensions, bool PatchBased>
void FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateDistributions()
{
    updateWeights();
}

template<int Dimensions, bool PatchBased>
void FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateParameters(const ArrayXXd &posteriorProbabilities)
{
    updateCoefficients(posteriorProbabilities);
    updateVariances();
}

template<int Dimensions, bool PatchBased>
double FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::stdQuantitativeChi2Test(const int index1, const int index2, const int component) const
{
    if (index1 == index2)
        return 0;
    
    const double diff = this->m_coefficients(index1, component) - this->m_coefficients(index2, component);
    return (diff * diff) / (double)(2.0 * this->m_chi2Sigma(component));
}

template<int Dimensions, bool PatchBased>
double FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::stdQuantitativeChi2TestPatch(const int index1, const int index2, const int component) const
{
    if (index1 == index2)
        return 0;
    
    double z = 0;
    for (int n = 0; n < this->m_patchSize; ++n)
    {
        const double diff = this->m_coefficients(this->m_patches(index1, n), component) - this->m_coefficients(this->m_patches(index2, n), component);
        z += (diff * diff);
    }
    return z / (double) (2.0 * this->m_chi2Sigma(component) * this->m_patchSize);
}

template<int Dimensions, bool PatchBased>
void FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateWeights()
{
#pragma omp parallel for
    for (int i = 0; i < this->m_nodes; ++i)
    {
        for (int j = 0; j < this->m_classes; ++j)
        {
            for (int d = 0; d < this->m_directions; ++d)
            {
                for (int n = 0; n < this->m_cliques; ++n)
                {
                    double z = 0;
                    if (PatchBased)
                        z = stdQuantitativeChi2TestPatch(i, this->neighbour(i, d, n), j);
                    else
                        z = stdQuantitativeChi2Test(i, this->neighbour(i, d, n), j);
                    z = z < SVFMM_CHI2_DEFAULT_MIN_Z ? SVFMM_CHI2_DEFAULT_MIN_Z : z;
                    z = z > SVFMM_CHI2_DEFAULT_MAX_Z ? SVFMM_CHI2_DEFAULT_MAX_Z : z;
                    this->m_weights(i, d, n, j) = (float) this->chi2pdf(z);
                }
            }
        }
    }
}

template<int Dimensions, bool PatchBased>
void FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateCoefficients(const ArrayXXd &posteriorProbabilities)
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
                double weightsNi = 0;
                double coeffsNi = 0;
                for (int n = 0; n < this->m_cliques; ++n)
                {
                    coeffsNi += (this->m_coefficients(this->neighbour(i, d, n), j) * (double) this->m_weights(i, d, n, j));
                    weightsNi += (double) this->m_weights(i, d, n, j);
                }
                coeffsNid += (coeffsNi / this->m_sigma(j, d));
                weightsNid += (weightsNi / this->m_sigma(j, d));
            }

            const double a = 1.0;
            const double b = -(coeffsNid / weightsNid);
            const double c = -(posteriorProbabilities(i, j) / (2.0 * weightsNid));
            double r1, r2;

            if (this->cuadraticSolver(a, b, c, r1, r2))
                coeffs_i(j) = r1;
            else
                coeffs_i(j) = SVFMM_ZERO;
        }
        newCoefficients.row(i) = this->conjugateProjection(coeffs_i, weights);
    }
    this->m_coefficients = newCoefficients;
}

template<int Dimensions, bool PatchBased>
void FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateVariances()
{
#pragma omp parallel for
    for (int j = 0; j < this->m_classes; ++j)
    {
        for (int d = 0; d < this->m_directions; ++d)
        {
            double DNormal = 0;
            for (int i = 0; i < this->m_nodes; ++i)
            {
                for (int n = 0; n < this->m_cliques; ++n)
                {
                    const double diff = this->m_coefficients(i, j) - this->m_coefficients(this->neighbour(i, d, n), j);
                    DNormal += ((diff * diff) * (double) this->m_weights(i, d, n, j));
                }
            }
            this->m_sigma(j, d) = DNormal / (double) (this->m_nodes * this->m_cliques);
        }
        
        double DChi2 = 0;
        for (int i = 0; i < this->m_nodes; ++i)
        {
            for (int n = 0; n < this->m_patchSize; ++n)
            {
                const int ni = this->m_patches(i, n);
                if (PatchBased)
                {
                    double patchDiff = 0;
                    for (int p = 0; p < this->m_patchSize; ++p)
                    {
                        const double diff = this->m_coefficients(this->m_patches(i, p), j) - this->m_coefficients(this->m_patches(ni, p), j);
                        patchDiff = (diff * diff);
                    }
                    DChi2 += (patchDiff / (double) this->m_patchSize);
                }
                else
                {
                    const double diff = this->m_coefficients(i, j) - this->m_coefficients(ni, j);
                    DChi2 += (diff * diff);
                }
            }
        }
        this->m_chi2Sigma(j) = DChi2 / (double) (this->m_nodes * this->m_patchSize);
    }
}

template<int Dimensions, bool PatchBased>
double FrequentistAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::logLikelihood()
{
    const double NM = (double) (this->m_nodes * this->m_cliques);
    const double NKDM = (double) (this->m_nodes * this->m_classes * this->m_directions * this->m_cliques);
    const double NKDMLog2PIHalf = NKDM * 0.918938533204673;
    
    ArrayXd logLikelihood(this->m_classes);
#pragma omp parallel for
    for (int j = 0; j < this->m_classes; ++j)
    {
        double sumLogLikelihoodPerDirection = 0;
        for (int d = 0; d < this->m_directions; ++d)
        {
            double D = 0;
            double sumLogU = 0;
            for (int i = 0; i < this->m_nodes; ++i)
            {
                for (int n = 0; n < this->m_cliques; ++n)
                {
                    const double u = (double) this->m_weights(i, d, n, j);
                    const double diff = this->m_coefficients(i, j) - this->m_coefficients(this->neighbour(i, d, n), j);
                    D += (diff * diff * u);
                    sumLogU += std::log(u);
                }
            }
            sumLogLikelihoodPerDirection += -(D / (2.0 * this->m_sigma(j, d))) + (sumLogU / 2.0) - (NM * (std::log(this->m_sigma(j, d)) / 2.0));
        }
        logLikelihood(j) = sumLogLikelihoodPerDirection;
    }
    return logLikelihood.sum() - NKDMLog2PIHalf;
}

#endif