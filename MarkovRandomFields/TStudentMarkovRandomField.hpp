/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* t-Student Markov Random Field                                            *
***************************************************************************/

#ifndef TSTUDENTMARKOVRANDOMFIELD_HPP
#define TSTUDENTMARKOVRANDOMFIELD_HPP

#define cimg_display 0
#include <CImg.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <boost/math/tools/roots.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include "MarkovRandomField.hpp"
#include "../Constants.hpp"

using namespace Eigen;

/***************************** t-Student Markov Random Field *****************************/

template<int Dimensions>
class TStudentMarkovRandomField : public virtual MarkovRandomFieldBase
{
public:
    virtual ~TStudentMarkovRandomField() {}
    
    Model model() const { return TSTUDENT; }
    
    TStudentMarkovRandomField<Dimensions>& operator = (const TStudentMarkovRandomField<Dimensions> &other) {}
    
protected:
    TStudentMarkovRandomField() {}
    TStudentMarkovRandomField(const TStudentMarkovRandomField<Dimensions> &other) {}

    virtual void updateFreedomDegrees() = 0;
    
    template <class T>
    class degreesFreedomFunctor
    {
    public:
        degreesFreedomFunctor(const double constant): m_constant(constant) {}
        std::pair<T, T> operator()(const T &nu)
        {
            T fx = std::log(nu / 2.0) - boost::math::digamma(nu / 2.0) + m_constant;
            T dx = (1.0 / nu) - (boost::math::trigamma(nu / 2.0) / 2.0);
            return std::make_pair(fx, dx);
        }
    private:
        double m_constant;
    };
};

/***************************** Isotropic t-Student Markov Random Field *****************************/

template<int Dimensions>
class IsotropicTStudentMarkovRandomField : public IsotropicMarkovRandomField<Dimensions>, public TStudentMarkovRandomField<Dimensions>
{
public:
    virtual ~IsotropicTStudentMarkovRandomField() {}
    
    ArrayXd sigma() const { return m_sigma; }
    Tensor<float, 3> weights() const { return m_weights; }
    Tensor<float, 3> logWeights() const { return m_logWeights; }
    ArrayXd nu() const { return m_nu; }
    
    IsotropicTStudentMarkovRandomField<Dimensions>& operator = (const IsotropicTStudentMarkovRandomField<Dimensions> &other);
    
protected:
    IsotropicTStudentMarkovRandomField();
    IsotropicTStudentMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    IsotropicTStudentMarkovRandomField(const IsotropicTStudentMarkovRandomField<Dimensions> &other);

    void updateFreedomDegrees();
    
    ArrayXd m_sigma;
    Tensor<float, 3> m_weights;
    Tensor<float, 3> m_logWeights;
    ArrayXd m_nu;
};

/***************************** Implementation *****************************/

template<int Dimensions>
IsotropicTStudentMarkovRandomField<Dimensions>::IsotropicTStudentMarkovRandomField()
: IsotropicMarkovRandomField<Dimensions>(), TStudentMarkovRandomField<Dimensions>(), m_sigma(ArrayXd()), m_weights(Tensor<float, 3>()), m_logWeights(Tensor<float, 3>()), m_nu(ArrayXd())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
IsotropicTStudentMarkovRandomField<Dimensions>::IsotropicTStudentMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: IsotropicMarkovRandomField<Dimensions>(mask, classes, topology), TStudentMarkovRandomField<Dimensions>(), m_sigma(ArrayXd(classes)), m_weights(Tensor<float, 3>(this->m_nodes, this->m_cliques, classes)), m_logWeights(Tensor<float, 3>(this->m_nodes, this->m_cliques, classes)), m_nu(classes)
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
IsotropicTStudentMarkovRandomField<Dimensions>::IsotropicTStudentMarkovRandomField(const IsotropicTStudentMarkovRandomField<Dimensions> &other)
: IsotropicMarkovRandomField<Dimensions>(other), TStudentMarkovRandomField<Dimensions>(other), m_sigma(other.sigma()), m_weights(other.weights()), m_logWeights(other.logWeights()), m_nu(other.nu())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
IsotropicTStudentMarkovRandomField<Dimensions>& IsotropicTStudentMarkovRandomField<Dimensions>::operator = (const IsotropicTStudentMarkovRandomField<Dimensions> &other)
{
    IsotropicMarkovRandomField<Dimensions>::operator=(other);
    TStudentMarkovRandomField<Dimensions>::operator=(other);
    
    m_sigma = other.sigma();
    m_weights = other.weights();
    m_logWeights = other.logWeights();
    m_nu = other.nu();
    
    return *this;
}

template<int Dimensions>
void IsotropicTStudentMarkovRandomField<Dimensions>::updateFreedomDegrees()
{
    const int nodes   = int(m_weights.dimension(0));
    const int cliques = int(m_weights.dimension(1));
    const int classes = int(m_weights.dimension(2));
    
    #pragma omp parallel for
    for (int j = 0; j < classes; ++j)
    {
        double sumWeights = 0;
        double sumLogWeights = 0;
        for (int i = 0; i < nodes; ++i)
        {
            for (int n = 0; n < cliques; ++n)
            {
                sumWeights += (double) m_weights(i, n, j);
                sumLogWeights += (double) m_logWeights(i, n, j);
            }
        }
        boost::uintmax_t maxIterations = 500;
        const double constant = ((sumLogWeights - sumWeights) / (double) (nodes * cliques)) + 1.0;
        typename TStudentMarkovRandomField<Dimensions>::template degreesFreedomFunctor<double> functor(constant);
        m_nu(j) = boost::math::tools::newton_raphson_iterate(functor, m_nu(j), 0.001, 50.0, static_cast<int>(std::numeric_limits<double>::digits * 0.6), maxIterations);
    }
}

/***************************** Anisotropic t-Student Markov Random Field *****************************/

template<int Dimensions>
class AnisotropicTStudentMarkovRandomField : public AnisotropicMarkovRandomField<Dimensions>, public TStudentMarkovRandomField<Dimensions>
{
public:
    virtual ~AnisotropicTStudentMarkovRandomField() {}
    
    ArrayXXd sigma() const { return m_sigma; }
    Tensor<float, 4> weights() const { return m_weights; }
    Tensor<float, 4> logWeights() const { return m_logWeights; }
    ArrayXXd nu() const { return m_nu; }
    
    AnisotropicTStudentMarkovRandomField<Dimensions>& operator = (const AnisotropicTStudentMarkovRandomField<Dimensions> &other);
    
protected:
    AnisotropicTStudentMarkovRandomField();
    AnisotropicTStudentMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    AnisotropicTStudentMarkovRandomField(const AnisotropicTStudentMarkovRandomField<Dimensions> &other);

    void updateFreedomDegrees();
    
    ArrayXXd m_sigma;
    Tensor<float, 4> m_weights;
    Tensor<float, 4> m_logWeights;
    ArrayXXd m_nu;
};

/***************************** Implementation *****************************/

template<int Dimensions>
AnisotropicTStudentMarkovRandomField<Dimensions>::AnisotropicTStudentMarkovRandomField()
: AnisotropicMarkovRandomField<Dimensions>(), TStudentMarkovRandomField<Dimensions>(), m_sigma(ArrayXXd()), m_weights(Tensor<float, 4>()), m_logWeights(Tensor<float, 4>()), m_nu(ArrayXXd())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
AnisotropicTStudentMarkovRandomField<Dimensions>::AnisotropicTStudentMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: AnisotropicMarkovRandomField<Dimensions>(mask, classes, topology), TStudentMarkovRandomField<Dimensions>(), m_sigma(ArrayXXd(classes, this->m_directions)), m_weights(Tensor<float, 4>(this->m_nodes, this->m_directions, this->m_cliques, classes)), m_logWeights(Tensor<float, 4>(this->m_nodes, this->m_directions, this->m_cliques, classes)), m_nu(classes, this->m_directions)
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
AnisotropicTStudentMarkovRandomField<Dimensions>::AnisotropicTStudentMarkovRandomField(const AnisotropicTStudentMarkovRandomField<Dimensions> &other)
: AnisotropicMarkovRandomField<Dimensions>(other), TStudentMarkovRandomField<Dimensions>(other), m_sigma(other.sigma()), m_weights(other.weights()), m_logWeights(other.logWeights()), m_nu(other.nu())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
AnisotropicTStudentMarkovRandomField<Dimensions>& AnisotropicTStudentMarkovRandomField<Dimensions>::operator = (const AnisotropicTStudentMarkovRandomField<Dimensions> &other)
{
    AnisotropicMarkovRandomField<Dimensions>::operator=(other);
    TStudentMarkovRandomField<Dimensions>::operator=(other);
    
    m_sigma = other.sigma();
    m_weights = other.weights();
    m_logWeights = other.logWeights();
    m_nu = other.nu();
    
    return *this;
}

template<int Dimensions>
void AnisotropicTStudentMarkovRandomField<Dimensions>::updateFreedomDegrees()
{
    const int nodes      = int(m_weights.dimension(0));
    const int directions = int(m_weights.dimension(1));
    const int cliques    = int(m_weights.dimension(2));
    const int classes    = int(m_weights.dimension(3));
    
    #pragma omp parallel for
    for (int d = 0; d < directions; ++d)
    {
        for (int j = 0; j < classes; ++j)
        {
            double sumWeights = 0;
            double sumLogWeights = 0;
            for (int i = 0; i < nodes; ++i)
            {
                for (int n = 0; n < cliques; ++n)
                {
                    sumWeights += (double) m_weights(i, d, n, j);
                    sumLogWeights += (double) m_logWeights(i, d, n, j);
                }
            }
            boost::uintmax_t maxIterations = 500;
            const double constant = ((sumLogWeights - sumWeights) / (double) (nodes * cliques)) + 1.0;
            typename TStudentMarkovRandomField<Dimensions>::template degreesFreedomFunctor<double> functor(constant);
            m_nu(j, d) = boost::math::tools::newton_raphson_iterate(functor, m_nu(j, d), 0.001, 50.0, static_cast<int>(std::numeric_limits<double>::digits * 0.6), maxIterations);
        }
    }
}

#endif