/*******************************************************************************
Copyright (c) 2010, Jonathan Hiller (Cornell University)
If used in publication cite "J. Hiller and H. Lipson "Dynamic Simulation of Soft Heterogeneous Objects" In press. (2011)"

This file is part of Voxelyze.
Voxelyze is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Voxelyze is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
See <http://www.opensource.org/licenses/lgpl-3.0.html> for license details.
*******************************************************************************/

#ifndef VX_MATERIALLINK_H
#define VX_MATERIALLINK_H

#include "VX_MaterialVoxel.h"

//!Defines the homogenous material properties of a link connecting two voxels.
/*!The constructor takes the two voxel materials and calculates the "best" material properties of a third material that captures the physical behavior. Beam constants are precomputed to quick access during simulation.

If the two input voxel materials are identical then this material is applied and beam constants are precomputed.

If the materials are different a third material is carefully crafted from the two and beam constants precomputed.
*/

class CVX_MaterialLink : public CVX_MaterialVoxel {
	public:
	CVX_MaterialLink(CVX_MaterialVoxel* mat1, CVX_MaterialVoxel* mat2); //!< Creates a link material from the two specified voxel materials. The order is unimportant. @param[in] mat1 voxel material on one side of the link. @param[in] mat2 voxel material on the other side of the link.
//	virtual ~CVX_MaterialLink(void); //!< Destructor
	CVX_MaterialLink(const CVX_MaterialLink& VIn) {*this = VIn;} //!< Copy constructor
	virtual CVX_MaterialLink& operator=(const CVX_MaterialLink& VIn); //!< Equals operator

protected:
	virtual bool updateAll(); // called whenever one or both constituent materials has changed (re-calculates this material).
	virtual bool updateDerived(); //updates all the derived quantities cache (based on density, size and elastic modulus

	CVX_MaterialVoxel *vox1Mat, *vox2Mat; //if a combined material, the two source materials

	float _a1, _a2, _b1, _b2, _b3; //for link beam force calculations
	float _sqA1, _sqA2xIp, _sqB1, _sqB2xFMp, _sqB3xIp; //for link beam damping calculations

	friend class CVoxelyze; //give the main simulation class full access
	friend class CVX_Link; //give links direct access to parameters
};


#endif //VX_MATERIALLINK_H