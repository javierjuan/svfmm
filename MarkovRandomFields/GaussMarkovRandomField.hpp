/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Gauss Markov Random Field                                             *
***************************************************************************/

#ifndef GAUSSMARKOVRANDOMFIELD_HPP
#define GAUSSMARKOVRANDOMFIELD_HPP

#define cimg_display 0
#include <CImg.h>
#include <Eigen/Dense>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "MarkovRandomField.hpp"
#include "../Constants.hpp"

using namespace Eigen;

/***************************** Gauss Markov Random Field *****************************/

template<int Dimensions>
class GaussMarkovRandomField : public virtual MarkovRandomFieldBase
{
public:
    virtual ~GaussMarkovRandomField() {}
    
    Model model() const { return GAUSS; }
    
    GaussMarkovRandomField<Dimensions>& operator = (const GaussMarkovRandomField<Dimensions> &other) {}
    
protected:
    GaussMarkovRandomField() {}
    GaussMarkovRandomField(const GaussMarkovRandomField<Dimensions> &other) {}
};

/***************************** Isotropic Gauss Markov Random Field *****************************/

template<int Dimensions>
class IsotropicGaussMarkovRandomField : public IsotropicMarkovRandomField<Dimensions>, public GaussMarkovRandomField<Dimensions>
{
public:
    virtual ~IsotropicGaussMarkovRandomField() {}
    
    ArrayXd sigma() const { return m_sigma; }
    
    IsotropicGaussMarkovRandomField<Dimensions>& operator = (const IsotropicGaussMarkovRandomField<Dimensions> &other);
    
protected:
    IsotropicGaussMarkovRandomField();
    IsotropicGaussMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    IsotropicGaussMarkovRandomField(const IsotropicGaussMarkovRandomField<Dimensions> &other);
    
    ArrayXd m_sigma;
};

/***************************** Implementation *****************************/

template<int Dimensions>
IsotropicGaussMarkovRandomField<Dimensions>::IsotropicGaussMarkovRandomField()
: IsotropicMarkovRandomField<Dimensions>(), GaussMarkovRandomField<Dimensions>(), m_sigma(ArrayXd())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
IsotropicGaussMarkovRandomField<Dimensions>::IsotropicGaussMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: IsotropicMarkovRandomField<Dimensions>(mask, classes, topology), GaussMarkovRandomField<Dimensions>(), m_sigma(ArrayXd(classes))
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
IsotropicGaussMarkovRandomField<Dimensions>::IsotropicGaussMarkovRandomField(const IsotropicGaussMarkovRandomField<Dimensions> &other)
: IsotropicMarkovRandomField<Dimensions>(other), GaussMarkovRandomField<Dimensions>(other), m_sigma(other.sigma())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
IsotropicGaussMarkovRandomField<Dimensions>& IsotropicGaussMarkovRandomField<Dimensions>::operator = (const IsotropicGaussMarkovRandomField<Dimensions> &other)
{
    IsotropicMarkovRandomField<Dimensions>::operator=(other);
    GaussMarkovRandomField<Dimensions>::operator=(other);
    
    m_sigma = other.sigma();
    
    return *this;
}

/***************************** Anisotropic Gauss Markov Random Field *****************************/

template<int Dimensions>
class AnisotropicGaussMarkovRandomField : public AnisotropicMarkovRandomField<Dimensions>, public GaussMarkovRandomField<Dimensions>
{
public:
    virtual ~AnisotropicGaussMarkovRandomField() {}
    
    ArrayXXd sigma() const { return m_sigma; }
    
    AnisotropicGaussMarkovRandomField<Dimensions>& operator = (const AnisotropicGaussMarkovRandomField<Dimensions> &other);
    
protected:
    AnisotropicGaussMarkovRandomField();
    AnisotropicGaussMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    AnisotropicGaussMarkovRandomField(const AnisotropicGaussMarkovRandomField<Dimensions> &other);
    
    ArrayXXd m_sigma;
};

/***************************** Implementation *****************************/

template<int Dimensions>
AnisotropicGaussMarkovRandomField<Dimensions>::AnisotropicGaussMarkovRandomField()
: AnisotropicMarkovRandomField<Dimensions>(), GaussMarkovRandomField<Dimensions>(), m_sigma(ArrayXXd())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
AnisotropicGaussMarkovRandomField<Dimensions>::AnisotropicGaussMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: AnisotropicMarkovRandomField<Dimensions>(mask, classes, topology), GaussMarkovRandomField<Dimensions>(), m_sigma(ArrayXXd(classes, this->m_directions))
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
AnisotropicGaussMarkovRandomField<Dimensions>::AnisotropicGaussMarkovRandomField(const AnisotropicGaussMarkovRandomField<Dimensions> &other)
: AnisotropicMarkovRandomField<Dimensions>(other), GaussMarkovRandomField<Dimensions>(other), m_sigma(other.sigma())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
AnisotropicGaussMarkovRandomField<Dimensions>& AnisotropicGaussMarkovRandomField<Dimensions>::operator = (const AnisotropicGaussMarkovRandomField<Dimensions> &other)
{
    AnisotropicMarkovRandomField<Dimensions>::operator=(other);
    GaussMarkovRandomField<Dimensions>::operator=(other);
    
    m_sigma = other.sigma();
    
    return *this;
}

#endif