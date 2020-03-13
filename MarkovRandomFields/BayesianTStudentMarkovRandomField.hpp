/*****************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                           */
/* Universidad Politecnica de Valencia, Spain                               */
/*                                                                          */
/* Copyright (C) 2020 Javier Juan Albarracin                                */
/*                                                                          */
/*****************************************************************************
* Bayesian t-Student Markov Random Fields                                    *
*****************************************************************************/

#ifndef BAYESIANTSTUDENTMARKOVRANDOMFIELD_HPP
#define BAYESIANTSTUDENTMARKOVRANDOMFIELD_HPP

#define cimg_display 0
#include <CImg.h>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <omp.h>
#include "MarkovRandomField.hpp"
#include "TStudentMarkovRandomField.hpp"
#include "../Constants.hpp"

using namespace Eigen;

/***************************** Bayesian Isotropic t-Student Markov Random Field *****************************/

template<int Dimensions>
class BayesianIsotropicTStudentMarkovRandomField : public IsotropicTStudentMarkovRandomField<Dimensions>, public BayesianMarkovRandomField<Dimensions>
{
public:
    BayesianIsotropicTStudentMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    BayesianIsotropicTStudentMarkovRandomField(const BayesianIsotropicTStudentMarkovRandomField<Dimensions> &other);

    std::unique_ptr<MarkovRandomField<Dimensions>> clone() const;
    
    BayesianIsotropicTStudentMarkovRandomField<Dimensions> & operator = (const BayesianIsotropicTStudentMarkovRandomField<Dimensions> &other);
    
    void initialization(const ArrayXXd &coefficients);
    void updateDistributions();
    void updateParameters(const ArrayXXd &posteriorProbabilities);
    double logLikelihood();

private:
    BayesianIsotropicTStudentMarkovRandomField();
    
    void updateWeights();
    void updateCoefficients(const ArrayXXd &posteriorProbabilities);
    void updateVariances();
};

/***************************** Implementation *****************************/

template<int Dimensions>
BayesianIsotropicTStudentMarkovRandomField<Dimensions>::BayesianIsotropicTStudentMarkovRandomField()
: IsotropicTStudentMarkovRandomField<Dimensions>(), BayesianMarkovRandomField<Dimensions>()
{
}

template<int Dimensions>
BayesianIsotropicTStudentMarkovRandomField<Dimensions>::BayesianIsotropicTStudentMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: IsotropicTStudentMarkovRandomField<Dimensions>(mask, classes, topology), BayesianMarkovRandomField<Dimensions>(this->m_coefficients)
{
}

template<int Dimensions>
BayesianIsotropicTStudentMarkovRandomField<Dimensions>::BayesianIsotropicTStudentMarkovRandomField(const BayesianIsotropicTStudentMarkovRandomField<Dimensions> &other)
: IsotropicTStudentMarkovRandomField<Dimensions>(other), BayesianMarkovRandomField<Dimensions>(other)
{
}

template<int Dimensions>
BayesianIsotropicTStudentMarkovRandomField<Dimensions>& BayesianIsotropicTStudentMarkovRandomField<Dimensions>::operator = (const BayesianIsotropicTStudentMarkovRandomField<Dimensions> &other)
{
    IsotropicTStudentMarkovRandomField<Dimensions>::operator=(other);
    BayesianMarkovRandomField<Dimensions>::operator=(other);
    
    return *this;
}

template<int Dimensions>
std::unique_ptr<MarkovRandomField<Dimensions>> BayesianIsotropicTStudentMarkovRandomField<Dimensions>::clone() const
{
    return std::unique_ptr<MarkovRandomField<Dimensions>>(new BayesianIsotropicTStudentMarkovRandomField<Dimensions>(*this));
}

template<int Dimensions>
void BayesianIsotropicTStudentMarkovRandomField<Dimensions>::initialization(const ArrayXXd &coefficients)
{
    if (coefficients.rows() != this->m_nodes && coefficients.cols() != this->m_classes)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent coefficients initialization" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    this->m_coefficients = coefficients;
    this->m_concentrations = coefficients;
    this->m_weights.setConstant(SVFMM_STUDENT_DEFAULT_NU);
    this->m_nu = ArrayXd::Constant(this->m_classes, SVFMM_STUDENT_DEFAULT_NU);
    updateVariances();
    updateWeights();
}

template<int Dimensions>
void BayesianIsotropicTStudentMarkovRandomField<Dimensions>::updateDistributions()
{
    updateWeights();
}

template<int Dimensions>
void BayesianIsotropicTStudentMarkovRandomField<Dimensions>::updateParameters(const ArrayXXd &posteriorProbabilities)
{
    updateCoefficients(posteriorProbabilities);
    updateVariances();
    this->updateFreedomDegrees();
}

template<int Dimensions>
void BayesianIsotropicTStudentMarkovRandomField<Dimensions>::updateWeights()
{
#pragma omp parallel for
    for (int i = 0; i < this->m_nodes; ++i)
    {
        for (int j = 0; j < this->m_classes; ++j)
        {
            for (int n = 0; n < this->m_cliques; ++n)
            {
                const double diff = this->m_concentrations(i, j) - this->m_concentrations(this->neighbour(i, n), j);
                const double z = (diff * diff) / this->m_sigma(j);
                const double num = (this->m_nu(j) + 1.0) / 2.0;
                const double den = (this->m_nu(j) + z) / 2.0;
                this->m_weights(i, n, j) = (float) (num / den);
                this->m_logWeights(i, n, j) = (float) boost::math::digamma(num) - std::log(den);
            }
        }
    }
}

template<int Dimensions>
void BayesianIsotropicTStudentMarkovRandomField<Dimensions>::updateCoefficients(const ArrayXXd &posteriorProbabilities)
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

template<int Dimensions>
void BayesianIsotropicTStudentMarkovRandomField<Dimensions>::updateVariances()
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
                D += ((diff * diff) * (double) this->m_weights(i, n, j));
            }
        }
        this->m_sigma(j) = D / (double) (this->m_nodes * this->m_cliques);
    }
}

template<int Dimensions>
double BayesianIsotropicTStudentMarkovRandomField<Dimensions>::logLikelihood()
{
    const double NM = (double) (this->m_nodes * this->m_cliques);
    const double NKMLog2PIHalf = (double) (this->m_nodes * this->m_classes * this->m_cliques) * 0.918938533204673;
    
    ArrayXd logLikelihood(this->m_classes);
#pragma omp parallel for
    for (int j = 0; j < this->m_classes; ++j)
    {
        const double nuHalf = this->m_nu(j) / 2.0;
        double D = 0;
        double sumLogU = 0;
        double sumLogUdU = 0;
        for (int i = 0; i < this->m_nodes; ++i)
        {
            for (int n = 0; n < this->m_cliques; ++n)
            {
                const double u = (double) this->m_weights(i, n, j);
                const double logU = (double) this->m_logWeights(i, n, j);
                const double diff = this->m_concentrations(i, j) - this->m_concentrations(this->neighbour(i, n), j);
                D += (diff * diff * u);
                sumLogU += logU;
                sumLogUdU += (logU - u);
            }
        }
        const double normalTerm = -(D / (2.0 * this->m_sigma(j))) + (sumLogU / 2.0) - (NM * (std::log(this->m_sigma(j)) / 2.0));
        const double gammaTerm = (NM * nuHalf * std::log(nuHalf)) - (NM * boost::math::lgamma(nuHalf)) + (nuHalf * sumLogUdU) - sumLogU;
        
        logLikelihood(j) = normalTerm + gammaTerm;
    }
    return logLikelihood.sum() - NKMLog2PIHalf;
}

/***************************** Bayesian Anisotropic t-Student Markov Random Field *****************************/

template<int Dimensions>
class BayesianAnisotropicTStudentMarkovRandomField : public AnisotropicTStudentMarkovRandomField<Dimensions>, public BayesianMarkovRandomField<Dimensions>
{
public:
    BayesianAnisotropicTStudentMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    BayesianAnisotropicTStudentMarkovRandomField(const BayesianAnisotropicTStudentMarkovRandomField<Dimensions> &other);

    std::unique_ptr<MarkovRandomField<Dimensions>> clone() const;
    
    BayesianAnisotropicTStudentMarkovRandomField<Dimensions> & operator = (const BayesianAnisotropicTStudentMarkovRandomField<Dimensions> &other);
    
    void initialization(const ArrayXXd &coefficients);
    void updateDistributions();
    void updateParameters(const ArrayXXd &posteriorProbabilities);
    double logLikelihood();

private:
    BayesianAnisotropicTStudentMarkovRandomField();
    
    void updateWeights();
    void updateCoefficients(const ArrayXXd &posteriorProbabilities);
    void updateVariances();
};

/***************************** Implementation *****************************/

template<int Dimensions>
BayesianAnisotropicTStudentMarkovRandomField<Dimensions>::BayesianAnisotropicTStudentMarkovRandomField()
: AnisotropicTStudentMarkovRandomField<Dimensions>(), BayesianMarkovRandomField<Dimensions>()
{
}

template<int Dimensions>
BayesianAnisotropicTStudentMarkovRandomField<Dimensions>::BayesianAnisotropicTStudentMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: AnisotropicTStudentMarkovRandomField<Dimensions>(mask, classes, topology), BayesianMarkovRandomField<Dimensions>(this->m_coefficients)
{
}

template<int Dimensions>
BayesianAnisotropicTStudentMarkovRandomField<Dimensions>::BayesianAnisotropicTStudentMarkovRandomField(const BayesianAnisotropicTStudentMarkovRandomField<Dimensions> &other)
: AnisotropicTStudentMarkovRandomField<Dimensions>(other), BayesianMarkovRandomField<Dimensions>(other)
{
}

template<int Dimensions>
BayesianAnisotropicTStudentMarkovRandomField<Dimensions>& BayesianAnisotropicTStudentMarkovRandomField<Dimensions>::operator = (const BayesianAnisotropicTStudentMarkovRandomField<Dimensions> &other)
{
    AnisotropicTStudentMarkovRandomField<Dimensions>::operator=(other);
    BayesianMarkovRandomField<Dimensions>::operator=(other);
    
    return *this;
}

template<int Dimensions>
std::unique_ptr<MarkovRandomField<Dimensions>> BayesianAnisotropicTStudentMarkovRandomField<Dimensions>::clone() const
{
    return std::unique_ptr<MarkovRandomField<Dimensions>>(new BayesianAnisotropicTStudentMarkovRandomField<Dimensions>(*this));
}

template<int Dimensions>
void BayesianAnisotropicTStudentMarkovRandomField<Dimensions>::initialization(const ArrayXXd &coefficients)
{
    if (coefficients.rows() != this->m_nodes && coefficients.cols() != this->m_classes)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent coefficients initialization" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    this->m_coefficients = coefficients;
    this->m_concentrations = coefficients;
    this->m_weights.setConstant(SVFMM_STUDENT_DEFAULT_NU);
    this->m_nu = ArrayXXd::Constant(this->m_classes, this->m_directions, SVFMM_STUDENT_DEFAULT_NU);
    updateVariances();
    updateWeights();
}

template<int Dimensions>
void BayesianAnisotropicTStudentMarkovRandomField<Dimensions>::updateDistributions()
{
    updateWeights();
}

template<int Dimensions>
void BayesianAnisotropicTStudentMarkovRandomField<Dimensions>::updateParameters(const ArrayXXd &posteriorProbabilities)
{
    updateCoefficients(posteriorProbabilities);
    updateVariances();
    this->updateFreedomDegrees();
}

template<int Dimensions>
void BayesianAnisotropicTStudentMarkovRandomField<Dimensions>::updateWeights()
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
                    const double diff = this->m_concentrations(i, j) - this->m_concentrations(this->neighbour(i, d, n), j);
                    const double z = (diff * diff) / this->m_sigma(j, d);
                    const double num = (this->m_nu(j, d) + 1.0) / 2.0;
                    const double den = (this->m_nu(j, d) + z) / 2.0;
                    this->m_weights(i, d, n, j) = (float) (num / den);
                    this->m_logWeights(i, d, n, j) = (float) (boost::math::digamma(num) - std::log(den));
                }
            }
        }
    }
}

template<int Dimensions>
void BayesianAnisotropicTStudentMarkovRandomField<Dimensions>::updateCoefficients(const ArrayXXd &posteriorProbabilities)
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

template<int Dimensions>
void BayesianAnisotropicTStudentMarkovRandomField<Dimensions>::updateVariances()
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
                    D += ((diff * diff) * (double) this->m_weights(i, d, n, j));
                }
            }
            this->m_sigma(j, d) = D / (double) (this->m_nodes * this->m_cliques);
        }
    }
}

template<int Dimensions>
double BayesianAnisotropicTStudentMarkovRandomField<Dimensions>::logLikelihood()
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
            const double nuHalf = this->m_nu(j, d) / 2.0;
            double D = 0;
            double sumLogU = 0;
            double sumLogUdU = 0;
            for (int i = 0; i < this->m_nodes; ++i)
            {
                for (int n = 0; n < this->m_cliques; ++n)
                {
                    const double u = (double) this->m_weights(i, d, n, j);
                    const double logU = (double) this->m_logWeights(i, d, n, j);
                    const double diff = this->m_concentrations(i, j) - this->m_concentrations(this->neighbour(i, d, n), j);
                    D += (diff * diff * u);
                    sumLogU += logU;
                    sumLogUdU += (logU - u);
                }
            }
            const double normalTerm = -(D / (2.0 * this->m_sigma(j, d))) + (sumLogU / 2.0) - (NM * (std::log(this->m_sigma(j, d)) / 2.0));
            const double gammaTerm = (NM * nuHalf * std::log(nuHalf)) - (NM * boost::math::lgamma(nuHalf)) + (nuHalf * sumLogUdU) - sumLogU;
            
            sumLogLikelihoodPerDirection += (normalTerm + gammaTerm);
        }
        logLikelihood(j) = sumLogLikelihoodPerDirection;
    }
    return logLikelihood.sum() - NKDMLog2PIHalf;
}

#endif