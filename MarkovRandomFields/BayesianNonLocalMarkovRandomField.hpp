/*****************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                           */
/* Universidad Politecnica de Valencia, Spain                               */
/*                                                                          */
/* Copyright (C) 2020 Javier Juan Albarracin                                */
/*                                                                          */
/*****************************************************************************
* Bayesian Non Local Markov Random Fields                                    *
*****************************************************************************/

#ifndef BAYESIANNONLOCALMARKOVRANDOMFIELD_HPP
#define BAYESIANNONLOCALMARKOVRANDOMFIELD_HPP

#define cimg_display 0
#include <CImg.h>
#include <omp.h>
#include "MarkovRandomField.hpp"
#include "NonLocalMarkovRandomField.hpp"
#include "../Constants.hpp"

using namespace Eigen;

/***************************** Bayesian Isotropic Non Local Markov Random Field *****************************/

template<int Dimensions, bool PatchBased = false>
class BayesianIsotropicNonLocalMarkovRandomField : public IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>, public BayesianMarkovRandomField<Dimensions>
{
public:
    BayesianIsotropicNonLocalMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    BayesianIsotropicNonLocalMarkovRandomField(const BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other);

    std::unique_ptr<MarkovRandomField<Dimensions>> clone() const;
    
    BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased> & operator = (const BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other);
    
    void initialization(const ArrayXXd &coefficients);
    void updateDistributions();
    void updateParameters(const ArrayXXd &posteriorProbabilities);
    double logLikelihood();

private:
    BayesianIsotropicNonLocalMarkovRandomField();
    
    void updateWeights();
    void updateCoefficients(const ArrayXXd &posteriorProbabilities);
    void updateVariances();
    
    inline double stdQuantitativeChi2Test(const int index1, const int index2, const int component) const;
    inline double stdQuantitativeChi2TestPatch(const int index1, const int index2, const int component) const;
};

/***************************** Implementation *****************************/

template<int Dimensions, bool PatchBased>
BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::BayesianIsotropicNonLocalMarkovRandomField()
: IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>(), BayesianMarkovRandomField<Dimensions>()
{
}

template<int Dimensions, bool PatchBased>
BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::BayesianIsotropicNonLocalMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>(mask, classes, topology), BayesianMarkovRandomField<Dimensions>(this->m_coefficients)
{
}

template<int Dimensions, bool PatchBased>
BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::BayesianIsotropicNonLocalMarkovRandomField(const BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other)
: IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>(other), BayesianMarkovRandomField<Dimensions>(other)
{
}

template<int Dimensions, bool PatchBased>
BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>& BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::operator = (const BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other)
{
    IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::operator=(other);
    BayesianMarkovRandomField<Dimensions>::operator=(other);
    
    return *this;
}

template<int Dimensions, bool PatchBased>
std::unique_ptr<MarkovRandomField<Dimensions>> BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::clone() const
{
    return std::unique_ptr<MarkovRandomField<Dimensions>>(new BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>(*this));
}

template<int Dimensions, bool PatchBased>
void BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::initialization(const ArrayXXd &coefficients)
{
    if (coefficients.rows() != this->m_nodes && coefficients.cols() != this->m_classes)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent coefficients initialization" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    this->m_coefficients = coefficients;
    this->m_concentrations = coefficients;
    this->m_weights.setConstant(this->chi2pdf(this->m_nu));
    updateVariances();
    updateWeights();
}

template<int Dimensions, bool PatchBased>
void BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateDistributions()
{
    updateWeights();
}

template<int Dimensions, bool PatchBased>
void BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateParameters(const ArrayXXd &posteriorProbabilities)
{
    updateCoefficients(posteriorProbabilities);
    updateVariances();
}

template<int Dimensions, bool PatchBased>
double BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::stdQuantitativeChi2Test(const int index1, const int index2, const int component) const
{
    if (index1 == index2)
        return 0;
    
    const double diff = this->m_concentrations(index1, component) - this->m_concentrations(index2, component);
    return (diff * diff) / (double)(2.0 * this->m_chi2Sigma(component));
}

template<int Dimensions, bool PatchBased>
double BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::stdQuantitativeChi2TestPatch(const int index1, const int index2, const int component) const
{
    if (index1 == index2)
        return 0;
    
    double z = 0;
    for (int n = 0; n < this->m_patchSize; ++n)
    {
        const double diff = this->m_concentrations(this->m_patches(index1, n), component) - this->m_concentrations(this->m_patches(index2, n), component);
        z += (diff * diff);
    }
    return z / (double) (2.0 * this->m_chi2Sigma(component) * this->m_patchSize);
}

template<int Dimensions, bool PatchBased>
void BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateWeights()
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
void BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateCoefficients(const ArrayXXd &posteriorProbabilities)
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
            double weightsNi = 0;
            for (int n = 0; n < this->m_cliques; ++n)
            {
                ANi += (this->m_concentrations(this->neighbour(i, n), j) * (double) this->m_weights(i, n, j));
                weightsNi += (double) this->m_weights(i, n, j);
            }

            const double a = 1.0;
            const double b = Ai_j - (ANi / weightsNi);
            const double c = -(Ai_j * ANi) / weightsNi;
            const double d = -(posteriorProbabilities(i, j) * Ai_j * this->m_sigma(j)) / (2.0 * weightsNi);
            
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

template<int Dimensions, bool PatchBased>
void BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateVariances()
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
                const double diff = this->m_concentrations(i, j) - this->m_concentrations(this->neighbour(i, n), j);
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
                        const double diff = this->m_concentrations(this->m_patches(i, p), j) - this->m_concentrations(this->m_patches(ni, p), j);
                        patchDiff = (diff * diff);
                    }
                    DChi2 += (patchDiff / (double) this->m_patchSize);
                }
                else
                {
                    const double diff = this->m_concentrations(i, j) - this->m_concentrations(ni, j);
                    DChi2 += (diff * diff);
                }
            }
        }
        this->m_sigma(j) = DNormal / (double) (this->m_nodes * this->m_cliques);
        this->m_chi2Sigma(j) = DChi2 / (double) (this->m_nodes * this->m_patchSize);
    }
}

template<int Dimensions, bool PatchBased>
double BayesianIsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::logLikelihood()
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
                const double diff = this->m_concentrations(i, j) - this->m_concentrations(this->neighbour(i, n), j);
                D += (diff * diff * u);
                sumLogU += std::log(u);
            }
        }
        logLikelihood(j) = -(D / (2.0 * this->m_sigma(j))) + (sumLogU / 2.0) - (NM * (std::log(this->m_sigma(j)) / 2.0));
    }
    return logLikelihood.sum() - NKMLog2PIHalf;
}

/***************************** Bayesian Anisotropic Non Local Markov Random Field *****************************/

template<int Dimensions, bool PatchBased = false>
class BayesianAnisotropicNonLocalMarkovRandomField : public AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>, BayesianMarkovRandomField<Dimensions>
{
public:
    BayesianAnisotropicNonLocalMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    BayesianAnisotropicNonLocalMarkovRandomField(const BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other);

    std::unique_ptr<MarkovRandomField<Dimensions>> clone() const;
    
    BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased> & operator = (const BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other);
    
    void initialization(const ArrayXXd &coefficients);
    void updateDistributions();
    void updateParameters(const ArrayXXd &posteriorProbabilities);
    double logLikelihood();

private:
    BayesianAnisotropicNonLocalMarkovRandomField();
    
    void updateWeights();
    void updateCoefficients(const ArrayXXd &posteriorProbabilities);
    void updateVariances();
    
    inline double stdQuantitativeChi2Test(const int index1, const int index2, const int component) const;
    inline double stdQuantitativeChi2TestPatch(const int index1, const int index2, const int component) const;
};

/***************************** Implementation *****************************/

template<int Dimensions, bool PatchBased>
BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::BayesianAnisotropicNonLocalMarkovRandomField()
: AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>(), BayesianMarkovRandomField<Dimensions>()
{
}

template<int Dimensions, bool PatchBased>
BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::BayesianAnisotropicNonLocalMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>(mask, classes, topology), BayesianMarkovRandomField<Dimensions>(this->m_coefficients)
{
}

template<int Dimensions, bool PatchBased>
BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::BayesianAnisotropicNonLocalMarkovRandomField(const BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other)
: AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>(other), BayesianMarkovRandomField<Dimensions>(other)
{
}

template<int Dimensions, bool PatchBased>
BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>& BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::operator = (const BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other)
{
    AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::operator=(other);
    BayesianMarkovRandomField<Dimensions>::operator=(other);
    
    return *this;
}

template<int Dimensions, bool PatchBased>
std::unique_ptr<MarkovRandomField<Dimensions>> BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::clone() const
{
    return std::unique_ptr<MarkovRandomField<Dimensions>>(new BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>(*this));
}

template<int Dimensions, bool PatchBased>
void BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::initialization(const ArrayXXd &coefficients)
{
    if (coefficients.rows() != this->m_nodes && coefficients.cols() != this->m_classes)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent coefficients initialization" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    this->m_coefficients = coefficients;
    this->m_concentrations = coefficients;
    this->m_weights.setConstant(this->chi2pdf(this->m_nu));
    updateVariances();
    updateWeights();
}

template<int Dimensions, bool PatchBased>
void BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateDistributions()
{
    updateWeights();
}

template<int Dimensions, bool PatchBased>
void BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateParameters(const ArrayXXd &posteriorProbabilities)
{
    updateCoefficients(posteriorProbabilities);
    updateVariances();
}

template<int Dimensions, bool PatchBased>
double BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::stdQuantitativeChi2Test(const int index1, const int index2, const int component) const
{
    if (index1 == index2)
        return 0;
    
    const double diff = this->m_concentrations(index1, component) - this->m_concentrations(index2, component);
    return (diff * diff) / (double)(2.0 * this->m_chi2Sigma(component));
}

template<int Dimensions, bool PatchBased>
double BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::stdQuantitativeChi2TestPatch(const int index1, const int index2, const int component) const
{
    if (index1 == index2)
        return 0;
    
    double z = 0;
    for (int n = 0; n < this->m_patchSize; ++n)
    {
        const double diff = this->m_concentrations(this->m_patches(index1, n), component) - this->m_concentrations(this->m_patches(index2, n), component);
        z += (diff * diff);
    }
    return z / (double) (2.0 * this->m_chi2Sigma(component) * this->m_patchSize);
}

template<int Dimensions, bool PatchBased>
void BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateWeights()
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
void BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateCoefficients(const ArrayXXd &posteriorProbabilities)
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
                double weightsNi = 0;
                double ANi = 0;
                for (int n = 0; n < this->m_cliques; ++n)
                {
                    ANi += (this->m_concentrations(this->neighbour(i, d, n), j) * (double) this->m_weights(i, d, n, j));
                    weightsNi += (double) this->m_weights(i, d, n, j);
                }

                ANid += (ANi / this->m_sigma(j, d));
                weightsNid += (weightsNi / this->m_sigma(j, d));
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

template<int Dimensions, bool PatchBased>
void BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::updateVariances()
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
                    const double diff = this->m_concentrations(i, j) - this->m_concentrations(this->neighbour(i, d, n), j);
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
                        const double diff = this->m_concentrations(this->m_patches(i, p), j) - this->m_concentrations(this->m_patches(ni, p), j);
                        patchDiff = (diff * diff);
                    }
                    DChi2 += (patchDiff / (double) this->m_patchSize);
                }
                else
                {
                    const double diff = this->m_concentrations(i, j) - this->m_concentrations(ni, j);
                    DChi2 += (diff * diff);
                }
            }
        }
        this->m_chi2Sigma(j) = DChi2 / (double) (this->m_nodes * this->m_patchSize);
    }
}

template<int Dimensions, bool PatchBased>
double BayesianAnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::logLikelihood()
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
                    const double diff = this->m_concentrations(i, j) - this->m_concentrations(this->neighbour(i, d, n), j);
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