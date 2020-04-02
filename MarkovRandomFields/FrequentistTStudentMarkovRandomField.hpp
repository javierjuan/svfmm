/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Frequentist t-Student Markov Random Fields                               *
***************************************************************************/

#ifndef FREQUENTISTTSTUDENTMARKOVRANDOMFIELD_HPP
#define FREQUENTISTTSTUDENTMARKOVRANDOMFIELD_HPP

#define cimg_display 0
#include <CImg.h>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <omp.h>
#include "MarkovRandomField.hpp"
#include "TStudentMarkovRandomField.hpp"
#include "../Constants.hpp"

using namespace Eigen;

/***************************** Frequentist Isotropic t-Student Markov Random Field *****************************/

template<int Dimensions>
class FrequentistIsotropicTStudentMarkovRandomField : public IsotropicTStudentMarkovRandomField<Dimensions>, public FrequentistMarkovRandomField<Dimensions>
{
public:
    FrequentistIsotropicTStudentMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    FrequentistIsotropicTStudentMarkovRandomField(const FrequentistIsotropicTStudentMarkovRandomField<Dimensions> &other);
    
    std::unique_ptr<MarkovRandomField<Dimensions>> clone() const;
    
    FrequentistIsotropicTStudentMarkovRandomField<Dimensions>& operator = (const FrequentistIsotropicTStudentMarkovRandomField<Dimensions> &other);
    
    void initialization(const ArrayXXd &coefficients);
    void updateDistributions();
    void updateParameters(const ArrayXXd &posteriorProbabilities);
    double logLikelihood();

private:
    FrequentistIsotropicTStudentMarkovRandomField();
    
    void updateWeights();
    void updateCoefficients(const ArrayXXd &posteriorProbabilities);
    void updateVariances();
};

/***************************** Implementation *****************************/

template<int Dimensions>
FrequentistIsotropicTStudentMarkovRandomField<Dimensions>::FrequentistIsotropicTStudentMarkovRandomField()
: IsotropicTStudentMarkovRandomField<Dimensions>(), FrequentistMarkovRandomField<Dimensions>()
{
}

template<int Dimensions>
FrequentistIsotropicTStudentMarkovRandomField<Dimensions>::FrequentistIsotropicTStudentMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: IsotropicTStudentMarkovRandomField<Dimensions>(mask, classes, topology), FrequentistMarkovRandomField<Dimensions>()
{
}

template<int Dimensions>
FrequentistIsotropicTStudentMarkovRandomField<Dimensions>::FrequentistIsotropicTStudentMarkovRandomField(const FrequentistIsotropicTStudentMarkovRandomField<Dimensions> &other)
: IsotropicTStudentMarkovRandomField<Dimensions>(other), FrequentistMarkovRandomField<Dimensions>(other)
{
}

template<int Dimensions>
FrequentistIsotropicTStudentMarkovRandomField<Dimensions>& FrequentistIsotropicTStudentMarkovRandomField<Dimensions>::operator = (const FrequentistIsotropicTStudentMarkovRandomField<Dimensions> &other)
{
    IsotropicTStudentMarkovRandomField<Dimensions>::operator=(other);
    FrequentistMarkovRandomField<Dimensions>::operator=(other);
    
    return *this;
}

template<int Dimensions>
std::unique_ptr<MarkovRandomField<Dimensions>> FrequentistIsotropicTStudentMarkovRandomField<Dimensions>::clone() const
{
    return std::unique_ptr<MarkovRandomField<Dimensions>>(new FrequentistIsotropicTStudentMarkovRandomField<Dimensions>(*this));
}

template<int Dimensions>
void FrequentistIsotropicTStudentMarkovRandomField<Dimensions>::initialization(const ArrayXXd &coefficients)
{
    if (coefficients.rows() != this->m_nodes && coefficients.cols() != this->m_classes)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent coefficients initialization" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    this->m_coefficients = coefficients;
    this->m_weights.setConstant(SVFMM_STUDENT_DEFAULT_NU);
    this->m_nu = ArrayXd::Constant(this->m_classes, SVFMM_STUDENT_DEFAULT_NU);
    updateVariances();
    updateWeights();
}

template<int Dimensions>
void FrequentistIsotropicTStudentMarkovRandomField<Dimensions>::updateDistributions()
{
    updateWeights();
}

template<int Dimensions>
void FrequentistIsotropicTStudentMarkovRandomField<Dimensions>::updateParameters(const ArrayXXd &posteriorProbabilities)
{
    updateCoefficients(posteriorProbabilities);
    updateVariances();
    this->updateFreedomDegrees();
}

template<int Dimensions>
void FrequentistIsotropicTStudentMarkovRandomField<Dimensions>::updateWeights()
{
    #pragma omp parallel for
    for (int i = 0; i < this->m_nodes; ++i)
    {
        for (int j = 0; j < this->m_classes; ++j)
        {
            for (int n = 0; n < this->m_cliques; ++n)
            {
                const double diff = this->m_coefficients(i, j) - this->m_coefficients(this->neighbour(i, n), j);
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
void FrequentistIsotropicTStudentMarkovRandomField<Dimensions>::updateCoefficients(const ArrayXXd &posteriorProbabilities)
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
                coeffs_i(j) = r1 > r2 ? r1 : r2;
            else
                coeffs_i(j) = SVFMM_ZERO;
        }
        newCoefficients.row(i) = this->conjugateProjection(coeffs_i, weights);
    }
    this->m_coefficients = newCoefficients;
}

template<int Dimensions>
void FrequentistIsotropicTStudentMarkovRandomField<Dimensions>::updateVariances()
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
                D += ((diff * diff) * (double) this->m_weights(i, n, j));
            }
        }
        this->m_sigma(j) = D / (double) (this->m_nodes * this->m_cliques);
    }
}

template<int Dimensions>
double FrequentistIsotropicTStudentMarkovRandomField<Dimensions>::logLikelihood()
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
                const double diff = this->m_coefficients(i, j) - this->m_coefficients(this->neighbour(i, n), j);
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

/***************************** Frequentist Anisotropic t-Student Markov Random Field *****************************/

template<int Dimensions>
class FrequentistAnisotropicTStudentMarkovRandomField : public AnisotropicTStudentMarkovRandomField<Dimensions>, public FrequentistMarkovRandomField<Dimensions>
{
public:
    FrequentistAnisotropicTStudentMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    FrequentistAnisotropicTStudentMarkovRandomField(const FrequentistAnisotropicTStudentMarkovRandomField<Dimensions> &other);

    std::unique_ptr<MarkovRandomField<Dimensions>> clone() const;
    
    FrequentistAnisotropicTStudentMarkovRandomField<Dimensions> & operator = (const FrequentistAnisotropicTStudentMarkovRandomField<Dimensions> &other);
    
    void initialization(const ArrayXXd &coefficients);
    void updateDistributions();
    void updateParameters(const ArrayXXd &posteriorProbabilities);
    double logLikelihood();

private:
    FrequentistAnisotropicTStudentMarkovRandomField();
    
    void updateWeights();
    void updateCoefficients(const ArrayXXd &posteriorProbabilities);
    void updateVariances();
};

/***************************** Implementation *****************************/

template<int Dimensions>
FrequentistAnisotropicTStudentMarkovRandomField<Dimensions>::FrequentistAnisotropicTStudentMarkovRandomField()
: AnisotropicTStudentMarkovRandomField<Dimensions>(), FrequentistMarkovRandomField<Dimensions>()
{
}

template<int Dimensions>
FrequentistAnisotropicTStudentMarkovRandomField<Dimensions>::FrequentistAnisotropicTStudentMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: AnisotropicTStudentMarkovRandomField<Dimensions>(mask, classes, topology), FrequentistMarkovRandomField<Dimensions>()
{
}

template<int Dimensions>
FrequentistAnisotropicTStudentMarkovRandomField<Dimensions>::FrequentistAnisotropicTStudentMarkovRandomField(const FrequentistAnisotropicTStudentMarkovRandomField<Dimensions> &other)
: AnisotropicTStudentMarkovRandomField<Dimensions>(other), FrequentistMarkovRandomField<Dimensions>(other)
{
}

template<int Dimensions>
FrequentistAnisotropicTStudentMarkovRandomField<Dimensions>& FrequentistAnisotropicTStudentMarkovRandomField<Dimensions>::operator = (const FrequentistAnisotropicTStudentMarkovRandomField<Dimensions> &other)
{
    AnisotropicTStudentMarkovRandomField<Dimensions>::operator=(other);
    FrequentistMarkovRandomField<Dimensions>::operator=(other);
    
    return *this;
}

template<int Dimensions>
std::unique_ptr<MarkovRandomField<Dimensions>> FrequentistAnisotropicTStudentMarkovRandomField<Dimensions>::clone() const
{
    return std::unique_ptr<MarkovRandomField<Dimensions>>(new FrequentistAnisotropicTStudentMarkovRandomField<Dimensions>(*this));
}

template<int Dimensions>
void FrequentistAnisotropicTStudentMarkovRandomField<Dimensions>::initialization(const ArrayXXd &coefficients)
{
    if (coefficients.rows() != this->m_nodes && coefficients.cols() != this->m_classes)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent coefficients initialization" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    this->m_coefficients = coefficients;
    this->m_weights.setConstant(SVFMM_STUDENT_DEFAULT_NU);
    this->m_nu = ArrayXXd::Constant(this->m_classes, this->m_directions, SVFMM_STUDENT_DEFAULT_NU);
    updateVariances();
    updateWeights();
}

template<int Dimensions>
void FrequentistAnisotropicTStudentMarkovRandomField<Dimensions>::updateDistributions()
{
    updateWeights();
}

template<int Dimensions>
void FrequentistAnisotropicTStudentMarkovRandomField<Dimensions>::updateParameters(const ArrayXXd &posteriorProbabilities)
{
    updateCoefficients(posteriorProbabilities);
    updateVariances();
    this->updateFreedomDegrees();
}

template<int Dimensions>
void FrequentistAnisotropicTStudentMarkovRandomField<Dimensions>::updateWeights()
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
                    const double diff = this->m_coefficients(i, j) - this->m_coefficients(this->neighbour(i, d, n), j);
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
void FrequentistAnisotropicTStudentMarkovRandomField<Dimensions>::updateCoefficients(const ArrayXXd &posteriorProbabilities)
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
                coeffs_i(j) = r1 > r2 ? r1 : r2;
            else
                coeffs_i(j) = SVFMM_ZERO;
        }
        newCoefficients.row(i) = this->conjugateProjection(coeffs_i, weights);
    }
    this->m_coefficients = newCoefficients;
}

template<int Dimensions>
void FrequentistAnisotropicTStudentMarkovRandomField<Dimensions>::updateVariances()
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
                    D += ((diff * diff) * (double) this->m_weights(i, d, n, j));
                }
            }
            this->m_sigma(j, d) = D / (double) (this->m_nodes * this->m_cliques);
        }
    }
}

template<int Dimensions>
double FrequentistAnisotropicTStudentMarkovRandomField<Dimensions>::logLikelihood()
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
                    const double diff = this->m_coefficients(i, j) - this->m_coefficients(this->neighbour(i, d, n), j);
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