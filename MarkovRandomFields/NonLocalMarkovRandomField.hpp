/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Non Local Markov Random Field                                            *
***************************************************************************/

#ifndef NONLOCALMARKOVRANDOMFIELD_HPP
#define NONLOCALMARKOVRANDOMFIELD_HPP

#define cimg_display 0
#include <CImg.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <boost/math/special_functions/trigamma.hpp>
#include "MarkovRandomField.hpp"
#include "../Constants.hpp"

using namespace Eigen;

/***************************** Non Local Markov Random Field *****************************/

template<int Dimensions, bool PatchBased = false>
class NonLocalMarkovRandomField : public virtual MarkovRandomFieldBase
{
public:
    virtual ~NonLocalMarkovRandomField() {}
    
    Model model() const { return NONLOCAL; }
    
    int nu() const { return m_nu; }
    int patchSize() const { return m_patchSize; }
    VectorXd chi2Sigma() const { return m_chi2Sigma; }
    ArrayXXi patches() const { return m_patches; }
    
    NonLocalMarkovRandomField<Dimensions, PatchBased>& operator = (const NonLocalMarkovRandomField<Dimensions, PatchBased> &other);
    
protected:
    NonLocalMarkovRandomField();
    NonLocalMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes);
    NonLocalMarkovRandomField(const NonLocalMarkovRandomField<Dimensions, PatchBased> &other);
    
    void build(const cimg_library::CImg<bool> &mask);
    double chi2pdf(const double z) const { return m_chi2Constant * (std::pow(z, (m_nu / 2.0) - 1.0) * std::exp(-z / 2.0)); }
    double chi2pdf(const double z, const int nu) const { return (1.0 / (std::pow(2.0, nu / 2.0) * boost::math::tgamma(nu / 2.0))) * (std::pow(z, (nu / 2.0) - 1.0) * std::exp(-z / 2.0)); }
    
    virtual inline double stdQuantitativeChi2Test(const int index1, const int index2, const int component) const = 0;
    virtual inline double stdQuantitativeChi2TestPatch(const int index1, const int index2, const int component) const = 0;
    
    int m_nu;
    int m_patchSize;
    double m_chi2Constant;
    VectorXd m_chi2Sigma;
    ArrayXXi m_patches;
};

/***************************** Implementation *****************************/

template<int Dimensions, bool PatchBased>
NonLocalMarkovRandomField<Dimensions, PatchBased>::NonLocalMarkovRandomField()
: m_nu(1), m_patchSize(Dimensions == 2 ? 9 : 27), m_chi2Sigma(VectorXd()), m_patches(ArrayXXi())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    m_chi2Constant = 1.0 / (std::pow(2.0, m_nu / 2.0) * boost::math::tgamma(m_nu / 2.0));
}

template<int Dimensions, bool PatchBased>
NonLocalMarkovRandomField<Dimensions, PatchBased>::NonLocalMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes)
: m_nu(1), m_patchSize(Dimensions == 2 ? 9 : 27), m_chi2Sigma(VectorXd(classes)), m_patches(ArrayXXi())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    m_chi2Constant = 1.0 / (std::pow(2.0, m_nu / 2.0) * boost::math::tgamma(m_nu / 2.0));
    build(mask);
}

template<int Dimensions, bool PatchBased>
NonLocalMarkovRandomField<Dimensions, PatchBased>::NonLocalMarkovRandomField(const NonLocalMarkovRandomField<Dimensions, PatchBased> &other)
: m_nu(other.nu()), m_patchSize(other.patchSize()), m_chi2Sigma(other.chi2Sigma()), m_patches(other.patches())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    m_chi2Constant = 1.0 / (std::pow(2.0, m_nu / 2.0) * boost::math::tgamma(m_nu / 2.0));
}

template<int Dimensions, bool PatchBased>
NonLocalMarkovRandomField<Dimensions, PatchBased>& NonLocalMarkovRandomField<Dimensions, PatchBased>::operator = (const NonLocalMarkovRandomField<Dimensions, PatchBased> &other)
{
    m_nu = other.nu();
    m_patchSize = other.patchSize();
    m_chi2Sigma = other.chi2Sigma();
    m_patches = other.patches();
    m_chi2Constant = 1.0 / (std::pow(2.0, m_nu / 2.0) * boost::math::tgamma(m_nu / 2.0));
    
    return *this;
}

template<int Dimensions, bool PatchBased>
void NonLocalMarkovRandomField<Dimensions, PatchBased>::build(const cimg_library::CImg<bool> &mask)
{
    const int width  = mask.width();
    const int height = mask.height();
    const int depth  = mask.depth();
    
    int nodes = 0;
    cimg_library::CImg<int> map(width, height, depth);
    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int z = 0; z < depth; ++z)
            {
                if (!mask(x, y, z))
                {
                    map(x, y, z) = -1;
                    continue;
                }

                map(x, y, z) = nodes;
                ++nodes;
            }
        }
    }
    
    m_patches.resize(nodes, Dimensions == 2 ? 9 : 27);
    
    int i = 0;
    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int z = 0; z < depth; ++z)
            {
                if (!mask(x, y, z))
                    continue;
                
                const int xm = x - 1 < 0       ? x : x - 1;
                const int xp = x + 1 >= width  ? x : x + 1;
                const int ym = y - 1 < 0       ? y : y - 1;
                const int yp = y + 1 >= height ? y : y + 1;

                m_patches(i, 0) = map(x, y, z);
                
                m_patches(i, 1) = mask(xm, y, z) ? map(xm, y, z) : map(x, y, z);
                m_patches(i, 2) = mask(xp, y, z) ? map(xp, y, z) : map(x, y, z);

                m_patches(i, 3) = mask(x, ym, z) ? map(x, ym, z) : map(x, y, z);
                m_patches(i, 4) = mask(x, yp, z) ? map(x, yp, z) : map(x, y, z);

                m_patches(i, 5) = mask(xm, ym, z) ? map(xm, ym, z) : map(x, y, z);
                m_patches(i, 6) = mask(xp, yp, z) ? map(xp, yp, z) : map(x, y, z);

                m_patches(i, 7) = mask(xp, ym, z) ? map(xp, ym, z) : map(x, y, z);
                m_patches(i, 8) = mask(xm, yp, z) ? map(xm, yp, z) : map(x, y, z);

                if (Dimensions == 3)
                {
                    const int zm = z - 1 < 0      ? z : z - 1;
                    const int zp = z + 1 >= depth ? z : z + 1;

                    m_patches(i, 9)  = mask(x, y, zm) ? map(x, y, zm) : map(x, y, z);
                    m_patches(i, 10) = mask(x, y, zp) ? map(x, y, zp) : map(x, y, z);

                    m_patches(i, 11) = mask(xm, y, zm) ? map(xm, y, zm) : map(x, y, z);
                    m_patches(i, 12) = mask(xp, y, zp) ? map(xp, y, zp) : map(x, y, z);

                    m_patches(i, 13) = mask(xm, ym, zm) ? map(xm, ym, zm) : map(x, y, z);
                    m_patches(i, 14) = mask(xp, yp, zp) ? map(xp, yp, zp) : map(x, y, z);

                    m_patches(i, 15) = mask(x, ym, zm) ? map(x, ym, zm) : map(x, y, z);
                    m_patches(i, 16) = mask(x, yp, zp) ? map(x, yp, zp) : map(x, y, z);

                    m_patches(i, 17) = mask(xp, ym, zm) ? map(xp, ym, zm) : map(x, y, z);
                    m_patches(i, 18) = mask(xm, yp, zp) ? map(xm, yp, zp) : map(x, y, z);

                    m_patches(i, 19) = mask(xp, y, zm) ? map(xp, y, zm) : map(x, y, z);
                    m_patches(i, 20) = mask(xm, y, zp) ? map(xm, y, zp) : map(x, y, z);

                    m_patches(i, 21) = mask(xp, yp, zm) ? map(xp, yp, zm) : map(x, y, z);
                    m_patches(i, 22) = mask(xm, ym, zp) ? map(xm, ym, zp) : map(x, y, z);

                    m_patches(i, 23) = mask(x, yp, zm) ? map(x, yp, zm) : map(x, y, z);
                    m_patches(i, 24) = mask(x, ym, zp) ? map(x, ym, zp) : map(x, y, z);

                    m_patches(i, 25) = mask(xm, yp, zm) ? map(xm, yp, zm) : map(x, y, z);
                    m_patches(i, 26) = mask(xp, ym, zp) ? map(xp, ym, zp) : map(x, y, z);
                }
                ++i;
            }
        }
    }
}

/***************************** Isotropic Non Local Markov Random Field *****************************/

template<int Dimensions, bool PatchBased = false>
class IsotropicNonLocalMarkovRandomField : public IsotropicMarkovRandomField<Dimensions>, public NonLocalMarkovRandomField<Dimensions, PatchBased>
{
public:
    virtual ~IsotropicNonLocalMarkovRandomField() {}
    
    ArrayXd sigma() const { return m_sigma; }
    Tensor<float, 3> weights() const { return m_weights; }
    ArrayXi overlappings() const { return m_overlappings; }
    
    IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>& operator = (const IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other);
    
protected:
    IsotropicNonLocalMarkovRandomField();
    IsotropicNonLocalMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    IsotropicNonLocalMarkovRandomField(const IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other);
    
    ArrayXd m_sigma;
    Tensor<float, 3> m_weights;
    ArrayXi m_overlappings;
};

/***************************** Implementation *****************************/

template<int Dimensions, bool PatchBased>
IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::IsotropicNonLocalMarkovRandomField()
: IsotropicMarkovRandomField<Dimensions>(), NonLocalMarkovRandomField<Dimensions, PatchBased>(), m_sigma(ArrayXd()), m_weights(Tensor<float, 3>()), m_overlappings(ArrayXi())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions, bool PatchBased>
IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::IsotropicNonLocalMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: IsotropicMarkovRandomField<Dimensions>(mask, classes, topology), NonLocalMarkovRandomField<Dimensions, PatchBased>(mask, classes), m_sigma(ArrayXd(classes)), m_weights(Tensor<float, 3>(this->m_patches.rows(), Dimensions == 2 ? 8 : 26, classes)), m_overlappings(ArrayXi(Dimensions == 2 ? 8 : 26))
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    // Pre-computation of overlappings given each neighbour
    if (Dimensions == 2)
    {
        if (topology == MarkovRandomFieldBase::ORTHOGONAL)
            m_overlappings << 6, 6, 6, 6;
        else
            m_overlappings << 6, 6, 6, 6, 4, 4, 4, 4;
    }
    else
    {
        if (topology == MarkovRandomFieldBase::ORTHOGONAL)
            m_overlappings << 18, 18, 18, 18, 18, 18;
        else
            m_overlappings << 18, 18, 18, 18, 12, 12, 12, 12, 18, 18, 12, 12, 8, 8, 12, 12, 8, 8, 12, 12, 8, 8, 12, 12, 8, 8;
    }
}

template<int Dimensions, bool PatchBased>
IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::IsotropicNonLocalMarkovRandomField(const IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other)
: IsotropicMarkovRandomField<Dimensions>(other), NonLocalMarkovRandomField<Dimensions, PatchBased>(other), m_sigma(other.sigma()), m_weights(other.weights()), m_overlappings(other.overlappings())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions, bool PatchBased>
IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>& IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::operator = (const IsotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other)
{
    IsotropicMarkovRandomField<Dimensions>::operator=(other);
    NonLocalMarkovRandomField<Dimensions, PatchBased>::operator=(other);
    
    m_sigma = other.sigma();
    m_weights = other.weights();
    m_overlappings = other.overlappings();
    
    return *this;
}

/***************************** Anisotropic Non Local Markov Random Field *****************************/

template<int Dimensions, bool PatchBased>
class AnisotropicNonLocalMarkovRandomField : public AnisotropicMarkovRandomField<Dimensions>, public NonLocalMarkovRandomField<Dimensions, PatchBased>
{
public:
    virtual ~AnisotropicNonLocalMarkovRandomField() {}
    
    ArrayXXd sigma() const { return m_sigma; }
    Tensor<float, 4> weights() const { return m_weights; }
    ArrayXXi overlappings() const { return m_overlappings; }
    
    AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>& operator = (const AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other);
    
protected:
    AnisotropicNonLocalMarkovRandomField();
    AnisotropicNonLocalMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    AnisotropicNonLocalMarkovRandomField(const AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other);
    
    ArrayXXd m_sigma;
    Tensor<float, 4> m_weights;
    ArrayXXi m_overlappings;
};

/***************************** Implementation *****************************/

template<int Dimensions, bool PatchBased>
AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::AnisotropicNonLocalMarkovRandomField()
: AnisotropicMarkovRandomField<Dimensions>(), NonLocalMarkovRandomField<Dimensions, PatchBased>(), m_sigma(ArrayXXd()), m_weights(Tensor<float, 4>()), m_overlappings(ArrayXXi())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions, bool PatchBased>
AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::AnisotropicNonLocalMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: AnisotropicMarkovRandomField<Dimensions>(mask, classes, topology), NonLocalMarkovRandomField<Dimensions, PatchBased>(mask, classes), m_sigma(ArrayXXd(classes, this->m_directions)), m_weights(Tensor<float, 4>(this->m_patches.rows(), Dimensions == 2 ? 4 : 13, 2, classes)), m_overlappings(ArrayXXi(Dimensions == 2 ? 4 : 13, 2))
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    // Pre-computation of overlappings given each neighbour
    if (Dimensions == 2)
    {
        if (topology == MarkovRandomFieldBase::ORTHOGONAL)
            m_overlappings << 6, 6, 6, 6;
        else
            m_overlappings << 6, 6, 6, 6, 4, 4, 4, 4;
    }
    else
    {
        if (topology == MarkovRandomFieldBase::ORTHOGONAL)
            m_overlappings << 18, 18, 18, 18, 18, 18;
        else
            m_overlappings << 18, 18, 18, 18, 12, 12, 12, 12, 18, 18, 12, 12, 8, 8, 12, 12, 8, 8, 12, 12, 8, 8, 12, 12, 8, 8;
    }
}

template<int Dimensions, bool PatchBased>
AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::AnisotropicNonLocalMarkovRandomField(const AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other)
: AnisotropicMarkovRandomField<Dimensions>(other), NonLocalMarkovRandomField<Dimensions, PatchBased>(other), m_sigma(other.sigma()), m_weights(other.weights()), m_overlappings(other.overlappings())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions, bool PatchBased>
AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>& AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased>::operator = (const AnisotropicNonLocalMarkovRandomField<Dimensions, PatchBased> &other)
{
    AnisotropicMarkovRandomField<Dimensions>::operator=(other);
    NonLocalMarkovRandomField<Dimensions, PatchBased>::operator=(other);
    
    m_sigma = other.sigma();
    m_weights = other.weights();
    m_overlappings = other.overlappings();
    
    return *this;
}

#endif