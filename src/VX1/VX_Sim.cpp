/*******************************************************************************
Copyright (c) 2010, Jonathan Hiller (Cornell University)
If used in publication cite "J. Hiller and H. Lipson "Dynamic Simulation of Soft Heterogeneous Objects" In press. (2011)"

This file is part of Voxelyze.
Voxelyze is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Voxelyze is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
See <http://www.opensource.org/licenses/lgpl-3.0.html> for license details.
*******************************************************************************/

#include "VX_Sim.h"
#include "VXS_Voxel.h"
#include "VXS_Bond.h"


#include <sstream>

#ifdef USE_OPEN_GL
#include "GL_Utils.h"
#endif

#ifdef VX2
#include "VX_Material.h"
#include "VX_Voxel.h"
#endif

CVX_Sim::CVX_Sim(void)// : VoxelInput(this), BondInput(this) // : out("Logfile.txt", std::ios::ate)
{
	ImportSurfMesh=NULL;
	MotionZeroed = false;

	CurSimFeatures = VXSFEAT_PLASTICITY | VXSFEAT_FAILURE;


	MixRadius=Vec3D<>(0.0, 0.0, 0.0);
	BlendModel=MB_LINEAR;
	PolyExp = 1.0;

	KinEHistory.resize(HISTORY_SIZE, -1.0);
	TotEHistory.resize(HISTORY_SIZE, -1.0);
	MaxMoveHistory.resize(HISTORY_SIZE, -1.0);

	SetStopConditionType();
	SetStopConditionValue();

	StatToCalc = CALCSTAT_ALL;

	ClearAll();
}

CVX_Sim::~CVX_Sim(void)
{
	ClearAll();

}

CVX_Sim& CVX_Sim::operator=(const CVX_Sim& rSim) //overload "=" 
{
	//TODO: set everything sensible equal.

	return *this;
}

void CVX_Sim::SaveVXAFile(std::string filename)
{
	CXML_Rip XML;
	WriteVXA(&XML);
	XML.SaveFile(filename);
}

bool CVX_Sim::LoadVXAFile(std::string filename, std::string* pRetMsg)
{
	CXML_Rip XML;
	if (!XML.LoadFile(filename, pRetMsg)) return false;
	ReadVXA(&XML, pRetMsg);
	return true;
}

void CVX_Sim::WriteVXA(CXML_Rip* pXML)
{
	pXML->DownLevel("VXA");
	pXML->SetElAttribute("Version", "1.1");
	WriteXML(pXML);
	pEnv->WriteXML(pXML);
	pEnv->pObj->WriteXML(pXML);
	pXML->UpLevel();
}

bool CVX_Sim::ReadVXA(CXML_Rip* pXML, std::string* RetMessage) //pointer to VXA element
{
//	pObj->ClearMatter();
	std::string ThisVersion = "1.1";
	std::string Version;
	pXML->GetElAttribute("Version", &Version);
	if (atof(Version.c_str()) > atof(ThisVersion.c_str())) if (RetMessage) *RetMessage += "Attempting to open newer version of VXA file. Results may be unpredictable.\nUpgrade to newest version of VoxCAD.\n";

	if (pXML->FindElement("Simulator")){
		ReadXML(pXML);
		pXML->UpLevel();
	}

	//load environment
	if (pEnv && pXML->FindElement("Environment")){
		pEnv->ReadXML(pXML);
		pXML->UpLevel();
	}

	//Load VXC if pObj is valid...
	if (pEnv->pObj && (pXML->FindElement("VXC") || pXML->FindElement("DMF"))){
		pEnv->pObj->ReadXML(pXML, false, RetMessage);
		pXML->UpLevel();
	}
	return true;
}

void CVX_Sim::WriteXML(CXML_Rip* pXML)
{
	pXML->DownLevel("Simulator");
		pXML->DownLevel("Integration");
		pXML->Element("Integrator", 0); //0 = euler in older versions
		pXML->Element("DtFrac", DtFrac);
		pXML->UpLevel();

		pXML->DownLevel("Damping");
		pXML->Element("BondDampingZ", BondDampingZ); //BondDampingZ);
		pXML->Element("ColDampingZ", ColDampingZ); //ColDampingZ);
		pXML->Element("SlowDampingZ", SlowDampingZ); //SlowDampingZ);
//		pXML->Element("BondDampingZ", Vx.material(0)->internalDamping()); //BondDampingZ);
//		pXML->Element("ColDampingZ", Vx.material(0)->collisionDamping()); //ColDampingZ);
//		pXML->Element("SlowDampingZ", Vx.material(0)->globalDamping()); //SlowDampingZ);
		pXML->UpLevel();

		pXML->DownLevel("Collisions");
		pXML->Element("SelfColEnabled", IsFeatureEnabled(VXSFEAT_COLLISIONS));
		pXML->Element("ColSystem", COL_SURFACE_HORIZON);
		pXML->Element("CollisionHorizon", 3.0);
		pXML->UpLevel();

		pXML->DownLevel("Features");
//		pXML->Element("MaxVelLimitEnabled", IsFeatureEnabled(VXSFEAT_MAX_VELOCITY));
//		pXML->Element("MaxVoxVelLimit", MaxVoxVelLimit);
		pXML->Element("BlendingEnabled", IsFeatureEnabled(VXSFEAT_BLENDING));
		pXML->Element("XMixRadius", MixRadius.x);
		pXML->Element("YMixRadius", MixRadius.y);
		pXML->Element("ZMixRadius", MixRadius.z);
		pXML->Element("BlendModel", BlendModel);
		pXML->Element("PolyExp", PolyExp);
//		pXML->Element("FluidDampEnabled", IsFeatureEnabled(VXSFEAT_));
		pXML->Element("VolumeEffectsEnabled", IsFeatureEnabled(VXSFEAT_VOLUME_EFFECTS));
//		pXML->Element("EnforceLatticeEnabled", IsFeatureEnabled(VXSFEAT_));
		pXML->UpLevel();

		pXML->DownLevel("StopCondition");
		pXML->Element("StopConditionType", (int)StopConditionType);
		pXML->Element("StopConditionValue", StopConditionValue);
		pXML->UpLevel();

		pXML->DownLevel("EquilibriumMode");
		pXML->Element("EquilibriumModeEnabled", IsFeatureEnabled(VXSFEAT_EQUILIBRIUM_MODE));
	//	pXML->Element("StopConditionValue", StopConditionValue);
		pXML->UpLevel();

		if (ImportSurfMesh){
			pXML->DownLevel("SurfMesh");
			ImportSurfMesh->WriteXML(pXML, true);
			pXML->UpLevel();
		}

//		WriteAdditionalSimXML(pXML);
	pXML->UpLevel();

}

bool CVX_Sim::ReadXML(CXML_Rip* pXML, std::string* RetMessage)
{
	int tmpInt;
	vfloat tmpVFloat;
	bool tmpBool;

	if (pXML->FindElement("Integration")){
//		if (pXML->FindLoadElement("Integrator", &tmpInt)) CurIntegrator = (IntegrationType)tmpInt; else CurIntegrator = I_EULER;
		if (!pXML->FindLoadElement("DtFrac", &DtFrac)) DtFrac = (vfloat)0.9;
		pXML->UpLevel();
	}
		
	if (pXML->FindElement("Damping")){
		float tmp;
		if (!pXML->FindLoadElement("BondDampingZ", &tmp)) SetBondDampZ(0.1); else SetBondDampZ(tmp); // BondDampingZ = 0.1;
		if (!pXML->FindLoadElement("ColDampingZ", &tmp)) SetCollisionDampZ(1.0); else SetCollisionDampZ(tmp); // ColDampingZ = 1.0;
		if (!pXML->FindLoadElement("SlowDampingZ", &tmp)) SetSlowDampZ(1.0); else SetSlowDampZ(tmp); //SlowDampingZ = 1.0;
		pXML->UpLevel();
	}

	if (pXML->FindElement("Collisions")){
		if (!pXML->FindLoadElement("SelfColEnabled", &tmpBool)) tmpBool=false; EnableFeature(VXSFEAT_COLLISIONS, tmpBool);
//		if (pXML->FindLoadElement("ColSystem", &tmpInt)) CurColSystem = (ColSystem)tmpInt; else CurColSystem = COL_SURFACE_HORIZON;
//		if (!pXML->FindLoadElement("CollisionHorizon", &CollisionHorizon)) CollisionHorizon = (vfloat)2.0;
		pXML->UpLevel();
	}

	if (pXML->FindElement("Features")){
		if (!pXML->FindLoadElement("MaxVelLimitEnabled", &tmpBool)) tmpBool = false; EnableFeature(VXSFEAT_MAX_VELOCITY, false); //EnableFeature(VXSFEAT_MAX_VELOCITY, tmpBool);
//		if (!pXML->FindLoadElement("MaxVoxVelLimit", &MaxVoxVelLimit)) MaxVoxVelLimit = (vfloat)0.1;
		if (!pXML->FindLoadElement("BlendingEnabled", &tmpBool)) tmpBool = false; EnableFeature(VXSFEAT_BLENDING, tmpBool);

		
		if (pXML->FindLoadElement("MixRadius", &tmpVFloat)) MixRadius = Vec3D<>(tmpVFloat, tmpVFloat, tmpVFloat); //look for legacy first
		else {
			if (!pXML->FindLoadElement("XMixRadius", &MixRadius.x)) MixRadius.x = 0;
			if (!pXML->FindLoadElement("YMixRadius", &MixRadius.y)) MixRadius.y = 0;
			if (!pXML->FindLoadElement("ZMixRadius", &MixRadius.z)) MixRadius.z = 0;
		}

		if (pXML->FindLoadElement("BlendModel", &tmpInt)) BlendModel = (MatBlendModel)tmpInt; else BlendModel = MB_LINEAR;
		if (!pXML->FindLoadElement("PolyExp", &PolyExp)) PolyExp = 1.0;

		if (!pXML->FindLoadElement("FluidDampEnabled", &tmpBool)) tmpBool = false; //do nothing for now...
		if (!pXML->FindLoadElement("VolumeEffectsEnabled", &tmpBool)) tmpBool = false; EnableFeature(VXSFEAT_VOLUME_EFFECTS, tmpBool);
		if (!pXML->FindLoadElement("EnforceLatticeEnabled", &tmpBool)) tmpBool = false;  //do nothing for now...
		pXML->UpLevel();
	}

	if (pXML->FindElement("StopCondition")){
		if (pXML->FindLoadElement("StopConditionType", &tmpInt)) SetStopConditionType((StopCondition)tmpInt); else SetStopConditionType();
		if (pXML->FindLoadElement("StopConditionValue", &tmpVFloat)) SetStopConditionValue(tmpVFloat); else SetStopConditionValue();
		pXML->UpLevel();
	}

	if (pXML->FindElement("EquilibriumMode")){
		if (!pXML->FindLoadElement("EquilibriumModeEnabled", &tmpBool)) tmpBool = false; if (tmpBool && !IsFeatureEnabled(VXSFEAT_EQUILIBRIUM_MODE)) EnableFeature(VXSFEAT_EQUILIBRIUM_MODE, true);
		//if (EquilibriumModeEnabled) EnableEquilibriumMode(true); //so it can set up energy history if necessary
		pXML->UpLevel();
	}
	
//	MeshAutoGenerated=true;
	if (pXML->FindElement("SurfMesh")){
		if (pXML->FindElement("CMesh")){
			if (!ImportSurfMesh) ImportSurfMesh = new CMesh;
			//MeshAutoGenerated=false;
			ImportSurfMesh->ReadXML(pXML);
			pXML->UpLevel();
		}
		pXML->UpLevel();
	}

	return true; //ReadAdditionalSimXML(pXML, RetMessage);
}


void CVX_Sim::ClearAll(void) //Reset all initialized variables
{
	Initalized = false;
	LocalVXC.ClearMatter();

	//This should be all the stuff set by "Import()"
//	VoxArray.clear();
//	BondArrayInternal.clear();
//	XtoSIndexMap.clear();
//	StoXIndexMap.clear();
//	SurfVoxels.clear();

//	MaxDispSinceLastBondUpdate = (vfloat)FLT_MAX; //arbitrarily high as a flag to populate bonds

	ClearHistories();

	dt = (vfloat)0.0; //calculated per-step
	CurTime = (vfloat)0.0;
	CurStepCount = 0;
//	DtFrozen = false;

	SS.Clear();
	IniCM = Vec3D<>(0,0,0);

	delete ImportSurfMesh;
	ImportSurfMesh=NULL;
//	SurfMesh.Clear();
//	MeshAutoGenerated=true;


}

void CVX_Sim::EnableFeature(const int VXSFEAT, bool Enabled)
{
#ifdef VX2
	if (Enabled) CurSimFeatures |= VXSFEAT;
	else CurSimFeatures &= ~VXSFEAT;

	switch (VXSFEAT){
	case VXSFEAT_GRAVITY: if(Enabled) Vx.setGravity(1.0); else Vx.setGravity(0.0); break;
	case VXSFEAT_FLOOR: Vx.enableFloor(Enabled); break;
	case VXSFEAT_COLLISIONS: Vx.enableCollisions(Enabled); break;
	case VXSFEAT_EQUILIBRIUM_MODE: EnableEquilibriumMode(Enabled); break;
	case VXSFEAT_TEMPERATURE: if (pEnv) pEnv->EnableTemp(Enabled); UpdateMatTemps(); break;
	case VXSFEAT_TEMPERATURE_VARY: if (pEnv) pEnv->EnableTempVary(Enabled); break;
	case VXSFEAT_VOLUME_EFFECTS: 
		EnableVolumeEffects(Enabled); break;

	}

#else
	if (Enabled) CurSimFeatures |= VXSFEAT;
	else CurSimFeatures &= ~VXSFEAT;

	//some specifics... ideally these could be put somewhere else to be more elegant.
	switch (VXSFEAT){
	case VXSFEAT_COLLISIONS: ColEnableChanged=true; break;
	case VXSFEAT_EQUILIBRIUM_MODE: EnableEquilibriumMode(Enabled); break;
	case VXSFEAT_VOLUME_EFFECTS: OptimalDt = CalcMaxDt(); break;

	case VXSFEAT_GRAVITY: if (pEnv) pEnv->EnableGravity(Enabled); break;
	case VXSFEAT_FLOOR: if (pEnv) pEnv->EnableFloor(Enabled); break;
	case VXSFEAT_TEMPERATURE: if (pEnv) pEnv->EnableTemp(Enabled); UpdateMatTemps(); break;
	case VXSFEAT_TEMPERATURE_VARY: if (pEnv) pEnv->EnableTempVary(Enabled); break;
	}
#endif
}

bool CVX_Sim::IsFeatureEnabled(const int VXSFEAT)
{
#ifdef VX2
	switch (VXSFEAT){
	case VXSFEAT_GRAVITY: return Vx.gravity() != 0.0f; break;
	case VXSFEAT_FLOOR: return Vx.isFloorEnabled(); break;
	case VXSFEAT_COLLISIONS: return Vx.isCollisionsEnabled(); break; //|| stuffity stuff
	}
	return (CurSimFeatures & VXSFEAT);

#else
	return (CurSimFeatures & VXSFEAT);
#endif
}


//bool CVX_Sim::UpdateAllVoxPointers() //updates all pointers into the VoxArray (call if reallocated!)
//{
//	//internal bonds
//	for (std::vector<CVXS_BondInternal>::iterator it = BondArrayInternal.begin(); it != BondArrayInternal.end(); it++){
//		if (!it->UpdateVoxelPtrs()) return false;
//	}
//	//collision bonds
//	for (std::vector<CVXS_BondCollision>::iterator it = BondArrayCollision.begin(); it != BondArrayCollision.end(); it++){
//		if (!it->UpdateVoxelPtrs()) return false;
//	}
//	return true;
//}

//int CVX_Sim::NumVox(void) const
//{
//	return (int)VoxArray.size(); //-1; //-1 is because we always have an input voxel shadow...
//}

//void CVX_Sim::UpdateAllBondPointers() //updates all pointers into the VoxArray (call if reallocated!)
//{
//	//Todo: evaluate if this is needed anymore?
//	for (std::vector<CVXS_Voxel>::iterator it = VoxArray.begin(); it != VoxArray.end(); it++){
////		if (!it->UpdateBondLinks()) return false;
//		it->UpdateColBondPointers();
//		it->UpdateInternalBondPtrs();
//	}
//}

//int CVX_Sim::NumBond(void) const
//{
//	return (int)BondArrayInternal.size();
//}
//
//int CVX_Sim::NumColBond(void) const
//{
//	return (int)BondArrayCollision.size();
//}

//void CVX_Sim::DeleteCollisionBonds(void)
//{
//	int iT = NumVox();
//	for (int i=0; i<iT; i++){
//		VoxArray[i].UnlinkColBonds();
//	}
//
//	BondArrayCollision.clear();
//
//}

//CVXS_Voxel* CVX_Sim::GetInputVoxel(void)
//{
//	return &VoxArray[InputVoxSInd];
//}
//
//CVXS_Bond* CVX_Sim::GetInputBond(void)
//{
//	return &BondArrayInternal[InputBondInd];
//}

/*! The environment should have been previously initialized and linked with a single voxel object. 
This function sets or resets the entire simulation with the new environment.
@param[in] pEnvIn A pointer to initialized CVX_Environment to import into the simulator.
@param[out] RetMessage A pointer to initialized string. Output information from the Import function is appended to this string.
*/
bool CVX_Sim::Import(CVX_Environment* pEnvIn, CMesh* pSurfMeshIn, std::string* RetMessage)
{
//#ifdef VX2
	if (pEnvIn != NULL) pEnv = pEnvIn;

	Initalized = false;
	Vx.clear();
	LocalVXC = *pEnv->pObj; //make a copy of the reference digital object!


	EnableFeature(VXSFEAT_GRAVITY, pEnv->IsGravityEnabled());
	SetGravityAccel(pEnv->GetGravityAccel());
	EnableFeature(VXSFEAT_FLOOR, pEnv->IsFloorEnabled());
	EnableFeature(VXSFEAT_TEMPERATURE, pEnv->IsTempEnabled());
	EnableFeature(VXSFEAT_TEMPERATURE_VARY, pEnv->IsTempVaryEnabled());



	//transfer the materials in (and map indices)...
	std::vector<CVX_Material*> VxcToVx2MatIndexMap; //size of VXC materials, index is temporary to VX2 mat
	VxcToVx2MatIndexMap.resize(LocalVXC.GetNumMaterials()-1, NULL); //skip erase
	muMemory.clear();
	for (int i=1; i<LocalVXC.GetNumMaterials(); i++){ //for each material
		if (LocalVXC.GetBaseMat(i)->GetMatType() == SINGLE) VxcToVx2MatIndexMap[i-1] = Vx.addMaterial();
		CopyMat(LocalVXC.GetBaseMat(i), VxcToVx2MatIndexMap[i-1]);
		VxcToVx2MatIndexMap[i-1]->setInternalDamping(BondDampingZ);
		VxcToVx2MatIndexMap[i-1]->setGlobalDamping(SlowDampingZ);
		VxcToVx2MatIndexMap[i-1]->setCollisionDamping(ColDampingZ);

		muMemory.push_back(LocalVXC.GetBaseMat(i)->GetPoissonsRatio()); //remember for toggleing volume effects on and off.

	}


	//add the voxels
	Vx.setVoxelSize(LocalVXC.GetLatDimEnv().x); 
	int x, y, z;
	std::vector<CVX_Voxel*> VoxList; //local list of all voxel pointers
	for (int i=0; i<LocalVXC.GetStArraySize(); i++){ //for each voxel in the array
		int VxcMatIndex = LocalVXC.GetMat(i)-1;
		if (VxcMatIndex>=0){
			LocalVXC.GetXYZNom(&x, &y, &z, i);
			VoxList.push_back(Vx.setVoxel(VxcToVx2MatIndexMap[VxcMatIndex], x, y, z));
		}
	}

	//set any boundary conditions (this can be optimized much better
	int NumBCs = pEnv->GetNumBCs();
	std::vector<int> Sizes(NumBCs, 0);
	for (int i=0; i<NumBCs; i++) Sizes[i] = pEnv->GetNumTouching(i); //count the number of voxels touching each bc
	Vec3D<> BCsize = pEnv->pObj->GetLatDimEnv()/2.0;
	Vec3D<> WSSize = pEnv->pObj->GetWorkSpace();
	int numVox = VoxList.size();
	for (int i=0; i<numVox; i++){
		CVX_Voxel* pThisVox = VoxList[i];
		Vec3D<> ThisPos = pThisVox->position()+LocalVXC.GetLatDimEnv()/2;
		for (int j = 0; j<NumBCs; j++){ //go through each primitive defined as a constraint!
			CVX_FRegion* pCurBc = pEnv->GetBC(j);
			char ThisDofFixed = pCurBc->DofFixed;
			if (pCurBc->GetRegion()->IsTouching(&ThisPos, &BCsize, &WSSize)){ //if this point is within
				if (IS_FIXED(DOF_X, ThisDofFixed)) pThisVox->external()->setDisplacement(X_TRANSLATE, pCurBc->Displace.x);
				if (IS_FIXED(DOF_Y, ThisDofFixed)) pThisVox->external()->setDisplacement(Y_TRANSLATE, pCurBc->Displace.y);
				if (IS_FIXED(DOF_Z, ThisDofFixed)) pThisVox->external()->setDisplacement(Z_TRANSLATE, pCurBc->Displace.z);
				if (IS_FIXED(DOF_TX, ThisDofFixed)) pThisVox->external()->setDisplacement(X_ROTATE, pCurBc->AngDisplace.x);
				if (IS_FIXED(DOF_TY, ThisDofFixed)) pThisVox->external()->setDisplacement(Y_ROTATE, pCurBc->AngDisplace.y);
				if (IS_FIXED(DOF_TZ, ThisDofFixed)) pThisVox->external()->setDisplacement(Z_ROTATE, pCurBc->AngDisplace.z);

				pThisVox->external()->addForce((Vec3D<float>)(pThisVox->external()->force() + pCurBc->Force/Sizes[j]));
				pThisVox->external()->addMoment((Vec3D<float>)(pThisVox->external()->moment() + pCurBc->Torque/Sizes[j]));
//				pThisVox->setExternalForce((Vec3D<float>)(pThisVox->externalForce() + pCurBc->Force/Sizes[j]));
//				pThisVox->setExternalMoment((Vec3D<float>)(pThisVox->externalMoment() + pCurBc->Torque/Sizes[j]));
			}
		}
	}

	//TMP!!
	//Vx.loadJSON("voxels.json");


	OptimalDt = Vx.recommendedTimeStep(); //to set up dialogs parameter ranges, we need this before the first iteration.
	if (IsFeatureEnabled(VXSFEAT_TEMPERATURE)) UpdateMatTemps();
	EnableVolumeEffects(IsFeatureEnabled(VXSFEAT_VOLUME_EFFECTS));

	Initalized = true;


//	Vx.saveJSON("test3.json"); //writeJSON();
//	Vx.loadJSON("test2.json"); //writeJSON();
//	Vx.doLinearSolve();
//	Vec3D<> tmp = Vx.voxel(2)->position();
//	int stop = 0;

//
//#else
//	ClearAll(); //clears out all arrays and stuff
//
//	if (pEnvIn != NULL) pEnv = pEnvIn;
//	if (pEnv == NULL) {if (RetMessage) *RetMessage += "Invalid Environment pointer"; return false;}
//
//	//get in sync with environment options
//	EnableFeature(VXSFEAT_GRAVITY, pEnv->IsGravityEnabled());
//	EnableFeature(VXSFEAT_FLOOR, pEnv->IsFloorEnabled());
//	EnableFeature(VXSFEAT_TEMPERATURE, pEnv->IsTempEnabled());
//	EnableFeature(VXSFEAT_TEMPERATURE_VARY, pEnv->IsTempVaryEnabled());
//
//	LocalVXC = *pEnv->pObj; //make a copy of the reference digital object!
//	if (LocalVXC.GetNumVox() == 0) {if (RetMessage) *RetMessage += "No voxels in object"; return false;}
//
//	int SIndexIt = 0; //keep track of how many voxel we've added (for storing reverse lookup array...)
//	int NumBCs = pEnv->GetNumBCs();
//	CVX_FRegion* pCurBc;
//
//
//	//initialize XtoSIndexMap & StoXIndexMap
//	XtoSIndexMap.resize(LocalVXC.GetStArraySize(), -1); // = new int[LocalVXC.GetStArraySize()];
//	StoXIndexMap.resize(LocalVXC.GetNumVox(), -1); // = new int [m_NumVox];
//
//
//	std::vector<int> Sizes(NumBCs, 0);
//	for (int i=0; i<NumBCs; i++) Sizes[i] = pEnv->GetNumTouching(i);
////	pEnv->GetNumVoxTouchingForced(&Sizes); //get the number of voxels in each region (to apply equal force to each voxel within this region!)
//
////	Vec3D BCpoint;
//	Vec3D<> BCsize = pEnv->pObj->GetLatDimEnv()/2.0;
//	Vec3D<> WSSize = pEnv->pObj->GetWorkSpace();
//
//	//Add all Voxels:
//	bool HasPlasticMaterial = false;
//	Vec3D<> ThisPos;
//	vfloat ThisScale = LocalVXC.GetLatDimEnv().x; //force to cubic
//	//Build voxel list
//	for (int i=0; i<LocalVXC.GetStArraySize(); i++){ //for each voxel in the array
//		XtoSIndexMap[i] = -1; //assume there is not a voxel here...
//
//		if(LocalVXC.Structure[i] != 0 ){ //if there's material here
//			int ThisMatIndex = LocalVXC.GetLeafMatIndex(i); 
//			int ThisMatModel = LocalVXC.Palette[ThisMatIndex].GetMatModel();
//			if (ThisMatModel == MDL_BILINEAR || ThisMatModel == MDL_DATA) HasPlasticMaterial = true; //enable plasticity in the sim
//
//			LocalVXC.GetXYZ(&ThisPos, i, false);//Get XYZ location
//
//			CVXS_Voxel CurVox(this, SIndexIt, i, ThisMatIndex, ThisPos, ThisScale);
//
//			XtoSIndexMap[i] = SIndexIt; //so we can find this voxel based on it's original index
//			StoXIndexMap[SIndexIt] = i; //so we can find the original index based on its simulator position
//			
//			for (int j = 0; j<NumBCs; j++){ //go through each primitive defined as a constraint!
//				pCurBc = pEnv->GetBC(j);
//				char ThisDofFixed = pCurBc->DofFixed;
//				if (pCurBc->GetRegion()->IsTouching(&ThisPos, &BCsize, &WSSize)){ //if this point is within
//					CurVox.FixDof(ThisDofFixed);
//					CurVox.AddExternalForce(pCurBc->Force/Sizes[j]);
//					CurVox.AddExternalTorque(pCurBc->Torque/Sizes[j]);
//
//					if (IS_FIXED(DOF_X, ThisDofFixed)) CurVox.SetExternalDisp(AXIS_X, pCurBc->Displace.x);
//					if (IS_FIXED(DOF_Y, ThisDofFixed)) CurVox.SetExternalDisp(AXIS_Y, pCurBc->Displace.y);
//					if (IS_FIXED(DOF_Z, ThisDofFixed)) CurVox.SetExternalDisp(AXIS_Z, pCurBc->Displace.z);
//					if (IS_FIXED(DOF_TX, ThisDofFixed)) CurVox.SetExternalTDisp(AXIS_X, pCurBc->AngDisplace.x);
//					if (IS_FIXED(DOF_TY, ThisDofFixed)) CurVox.SetExternalTDisp(AXIS_Y, pCurBc->AngDisplace.y);
//					if (IS_FIXED(DOF_TZ, ThisDofFixed)) CurVox.SetExternalTDisp(AXIS_Z, pCurBc->AngDisplace.z);
//
////					CurVox.SetExternalDisp(pCurBc->Displace);
////					CurVox.SetExternalTDisp(pCurBc->AngDisplace);
//				}
//			}
//
////			if(BlendingEnabled) CurVox.CalcMyBlendMix(); //needs to be done basically last. Todo next: move to constructor and ditch blendmix and even p_sim from voxel?
//			try{VoxArray.push_back(CurVox);}
//			catch (std::bad_alloc&){if (RetMessage) *RetMessage += "Insufficient memory. Reduce model size.\n"; return false;} //catch if we run out of memory
//
//			SIndexIt++;
//		}
//	}
//
//
//	//add input voxel so that NumVox() works!
////	InputVoxSInd = (int)VoxArray.size();
////	CVXS_Voxel TmpVox(this, 0, 0, 0, Vec3D<>(0,0,0), 0);
////	TmpVox.LinkToVXSim(this);
////	VoxArray.push_back(TmpVox);
//
//
//	//Set up all permanent bonds
//	//Between adjacent voxels in the lattice
//	int ThisX=0, ThisY=0, ThisZ=0, posXInd=0; //index of the nex voxel in positive directions
//	std::string BondFailMsg = "At least one bond creation failed during import.\n";
//	for (int i=0; i<NumVox(); i++){ //for each voxel in our newly-made array look in the +X, +Y and +Z directions to form a bond
//		LocalVXC.GetXYZNom(&ThisX, &ThisY, &ThisZ, StoXIndexMap[i]);
//
//		for (int j=0; j<3; j++){ //for each positive direction in the lattice
//			switch (j){ 
//				case 0: posXInd = LocalVXC.GetIndex(ThisX+1, ThisY, ThisZ); break; //X
//				case 1: posXInd = LocalVXC.GetIndex(ThisX, ThisY+1, ThisZ); break; //Y
//				case 2: posXInd = LocalVXC.GetIndex(ThisX, ThisY, ThisZ+1); break; //Z
//			}
//			if (posXInd != -1 && LocalVXC.Structure[posXInd]){
//				bool BondCreated;
//				try {BondCreated = CreatePermBond(i, XtoSIndexMap[posXInd]);}
//				catch (std::bad_alloc&){if (RetMessage) *RetMessage += "Insufficient memory. Reduce model size.\n"; return false;} //catch if we run out of memory
//
//				if(!BondCreated && RetMessage) *RetMessage += BondFailMsg; //warning if it wasn't a memory throw
//		
//			}
//		}
//	}
//
//	//Create input bond
////	CreateBond(B_INPUT_LINEAR_NOROT, InputVoxSInd, InputVoxSInd, true, &InputBondInd, false); //create input bond, but initialize it to meaningless connection to self
//
//	UpdateAllBondPointers(); //necessary since we probably reallocated the bond array when adding pbonds the first time
//
//	//Set up our surface list...
//	for (int i=0; i<NumVox(); i++){ //for each voxel in our newly-made array
//		if (VoxArray[i].IsSurfaceVoxel()){
////		if (VoxArray[i].GetNumLocalBonds() != 6){
//			try {SurfVoxels.push_back(i);}
//			catch (std::bad_alloc&){if (RetMessage) *RetMessage += "Insufficient memory. Reduce model size.\n"; return false;} //catch if we run out of memory
//
//		}
//
//		//todo: only do for those on surfaces, I think.
//		VoxArray[i].CalcNearby(this, (int)(CollisionHorizon*1.5)); //populate the nearby array
//	}
//
//	if (pSurfMeshIn){
//		if (!ImportSurfMesh) ImportSurfMesh = new CMesh;
//		*ImportSurfMesh = *pSurfMeshIn;
//
//	}
//
//
////ifdef USE_OPEN_GL
//	//VoxMesh.ImportLinkSim(this);
//	//VoxMesh.DefMesh.DrawSmooth = false;
//
//	////if the input mesh is not valid, use marching cubes to create one
//	//if (!pSurfMeshIn){
//	//	MeshAutoGenerated = true;
//	//	CMesh GeneratedSmoothMesh;
//
//	//	CArray3Df OccupancyArray(pEnv->pObj->GetVXDim(), pEnv->pObj->GetVYDim(), pEnv->pObj->GetVZDim()); 
//	//	int NumPossibleVox = pEnv->pObj->GetStArraySize();
//	//	for (int g=0; g<NumPossibleVox; g++){
//	//		if (pEnv->pObj->Structure.GetData(g)>0) OccupancyArray[g] = 1.0;
//	//	}
//	//	CMarchCube::SingleMaterial(&GeneratedSmoothMesh, &OccupancyArray, 0.5, pEnv->pObj->GetLatticeDim());
//	//	SurfMesh.ImportSimWithMesh(this, &GeneratedSmoothMesh);
//	//}
//	//else {
//	//	MeshAutoGenerated=false;
//	//	SurfMesh.ImportSimWithMesh(this, pSurfMeshIn);
//	//}
//
////#endif
//
//
//	ResetSimulation();
//	OptimalDt = CalcMaxDt(); //to set up dialogs parameter ranges, we need this before the first iteration.
//	EnableFeature(VXSFEAT_PLASTICITY, HasPlasticMaterial);
////	EnablePlasticity(HasPlasticMaterial); //turn off plasticity if we don't need it...
//
//	Initalized = true;
////	std::string tmpString;
//
//	std::ostringstream os;
//	os << "Completed Simulation Import: " << NumVox() << " Voxels, " << NumBond() << "Bonds.\n";
//	*RetMessage += os.str();
//
//#endif
	return true;
}

#ifdef VX2
void CVX_Sim::CopyMat(CVXC_Material* pOld, CVX_Material* pNew) //copies parameters from pOld to pNew
{
	pNew->setName(pOld->GetName().c_str());
	pNew->setColor(pOld->GetRedi(), pOld->GetGreeni(), pOld->GetBluei(), pOld->GetAlphai());
	switch (pOld->GetMatModel()){
	case MDL_LINEAR: pNew->setModelLinear(pOld->GetElasticMod()); break;
	case MDL_LINEAR_FAIL: pNew->setModelLinear(pOld->GetElasticMod(), pOld->GetFailStress()); break;
	case MDL_BILINEAR: pNew->setModelBilinear(pOld->GetElasticMod(), pOld->GetPlasticMod(), pOld->GetYieldStress(), pOld->GetFailStress()); break;
	case MDL_DATA: {
		std::vector<float> tmpStress, tmpStrain;
		int numPts=pOld->GetDataPointCount();
		for (int i=0; i<numPts; i++){
			tmpStress.push_back(pOld->GetStressData(i));
			tmpStrain.push_back(pOld->GetStrainData(i));
		}
		pNew->setModel(numPts, &(tmpStrain[0]), &(tmpStress[0])); break;
	}
	}

	pNew->setPoissonsRatio(pOld->GetPoissonsRatio());
	pNew->setDensity(pOld->GetDensity());
	pNew->setCte(pOld->GetCTE());
	pNew->setStaticFriction(pOld->GetuStatic());
	pNew->setKineticFriction(pOld->GetuDynamic());
	pNew->setGlobalDamping(GetSlowDampZ());
	pNew->setInternalDamping(GetBondDampZ());
	pNew->setCollisionDamping(GetCollisionDampZ());



}
#endif


/*! This bond is appended to the master bond array (BondArrayInternal). 
The behavior of the bond is determined by BondType. If the bond is permanent and should persist throughout the simulation PermIn should be to true.
@param[in] BondTypeIn The physical behavior of the bond being added.
@param[in] SIndex1 One simulation voxel index to be joined.
@param[in] SIndex2 The other simulation voxel index to be joined.
@param[in] PermIn Denotes whether this bond should persist throughout the simulation (true) or is temporary (false).
*/
//int CVX_Sim::CreatePermBond(int SIndexNegIn, int SIndexPosIn) //take care of all allocation, etc.
//{
//	if(IS_ALL_FIXED(VoxArray[SIndexNegIn].GetDofFixed()) && IS_ALL_FIXED(VoxArray[SIndexPosIn].GetDofFixed())) return -1; //if both voxels are fixed don't bother making a bond. (unnecessary)
//
//	CVXS_BondInternal tmp(this);
////	if (!tmp.DefineBond(B_LINEAR, SIndexNegIn, SIndexPosIn)) return -1;
//	if (!tmp.LinkVoxels(SIndexNegIn, SIndexPosIn)) return -1;
//
//	BondArrayInternal.push_back(tmp);
//
//
//	int MyBondIndex = NumBond()-1;
//
//	BondDir nVoxBD, pVoxBD;
//	switch (BondArrayInternal.back().GetBondAxis()){
//	case AXIS_X: nVoxBD=BD_PX; pVoxBD=BD_NX; break;
//	case AXIS_Y: nVoxBD=BD_PY; pVoxBD=BD_NY; break;
//	case AXIS_Z: nVoxBD=BD_PZ; pVoxBD=BD_NZ; break;
//	case AXIS_NONE: BondArrayInternal.pop_back(); return -1; //fail. can only deal with bond aligned with an axis here.
//	}
//
//	VoxArray[SIndexNegIn].LinkInternalBond(MyBondIndex, nVoxBD);
//	VoxArray[SIndexPosIn].LinkInternalBond(MyBondIndex, pVoxBD);
//
//	return MyBondIndex;
//}
//
//int CVX_Sim::CreateColBond(int SIndex1In, int SIndex2In) //!< Creates a new collision bond between two voxels. 
//{
//	CVXS_BondCollision tmp(this);
//	if (!tmp.LinkVoxels(SIndex1In, SIndex2In)) return -1;
////	if (!tmp.DefineBond(B_LINEAR_CONTACT, SIndex1In, SIndex2In)) return -1;
//
//	try {BondArrayCollision.push_back(tmp);}
//	catch (std::bad_alloc&){return -1;} //catch if we run out of memory
//
//	int MyBondIndex = NumColBond()-1;
//
//	VoxArray[SIndex1In].LinkColBond(MyBondIndex);
//	VoxArray[SIndex2In].LinkColBond(MyBondIndex);
//
//	return MyBondIndex;
//
//}

//bool CVX_Sim::CreateBond(BondType BondTypeIn, int SIndex1In, int SIndex2In, bool PermIn, int* pBondIndexOut, bool LinkBond) //take care of all dynamic memory stuff...
//{
//	if(IS_ALL_FIXED(VoxArray[SIndex1In].GetDofFixed()) && IS_ALL_FIXED(VoxArray[SIndex2In].GetDofFixed())) return true; //if both voxels are fixed don't bother making a bond. (unnecessary)
//
//	CVXS_Bond tmp(this);
//	if (!tmp.DefineBond(BondTypeIn, SIndex1In, SIndex2In, PermIn)) return false;
////	tmp.ThisBondType = BondTypeIn;
////	tmp.SetVox1SInd(SIndex1In);
////	tmp.SetVox2SInd(SIndex2In);
////	tmp.OrigDist = VoxArray[SIndex2In].GetNominalPosition() - VoxArray[SIndex1In].GetNominalPosition(); //was Pos
////	tmp.Perm = PermIn; 
//
////	tmp.UpdateConstants(); //do this here, cause we can...
//
//	BondArrayInternal.push_back(tmp);
//	
//	int MyBondIndex = NumBond()-1;
//
//	//if permanent bond
//	VoxArray[SIndex1In].InternalBondIndices[3] = MyBondIndex;
//	VoxArray[SIndex2In].InternalBondIndices[3] = MyBondIndex;
//
//
//	//else non-permanent bond
//
//
////	if(LinkBond){
////		VoxArray[SIndex1In].LinkBond(MyBondIndex);
////		VoxArray[SIndex2In].LinkBond(MyBondIndex);
////	}
//	if (pBondIndexOut) *pBondIndexOut = MyBondIndex;
//	return true;
//}

//bool CVX_Sim::UpdateBond(int BondIndex, int NewSIndex1In, int NewSIndex2In, bool LinkBond)
//{
//	CVXS_Bond* pThisBond = &BondArrayInternal[BondIndex];
//
//	pThisBond->DefineBond(pThisBond->GetBondType(), NewSIndex1In, NewSIndex2In, pThisBond->IsPermanent());
//	
//	//VoxArray[NewSIndex1In].LinkBond(BondIndex);
//	//VoxArray[NewSIndex2In].LinkBond(BondIndex);
//
//
//	//
////	if(LinkBond){ //unlink pointers to this bond from other voxels
////		VoxArray[pThisBond->GetVox1SInd()].UnlinkBond(BondIndex);
////		VoxArray[pThisBond->GetVox2SInd()].UnlinkBond(BondIndex);
////	}
////
////	pThisBond->DefineBond(pThisBond->GetBondType(), NewSIndex1In, NewSIndex2In, pThisBond->IsPermanent());
////	//set and linknew indices
//////	if (!pThisBond->SetVox1SInd(NewSIndex1In)) return false;
//////	if (!pThisBond->SetVox2SInd(NewSIndex2In)) return false;
////
//////	pThisBond->OrigDist = VoxArray[NewSIndex1In].GetNominalPosition() - VoxArray[NewSIndex2In].GetNominalPosition(); //Was Pos
//////	pThisBond->UpdateConstants(); //material may have changed with switch
//////	pThisBond->ResetBond(); //??
////
////	if(LinkBond){ //link the involved voxels to this bond
////		VoxArray[NewSIndex1In].LinkBond(BondIndex);
////		VoxArray[NewSIndex2In].LinkBond(BondIndex);
////	}
////
//	return true;
//}


void CVX_Sim::ResetSimulation(void)
{
#ifdef VX2
	Vx.resetTime();

	dt = (vfloat)0.0; //calculated per-step
	CurTime = (vfloat)0.0;
	CurStepCount = 0;
#else
	int iT = NumVox();
	for (int i=0; i<iT; i++) VoxArray[i].ResetVoxel();

	iT = NumBond();
	for (int j=0; j<iT; j++) BondArrayInternal[j].ResetBond();

	BondArrayCollision.clear();
//	if(SelfColEnabled) UpdateCollisions();
//	CalcL1Bonds(CollisionHorizon);
//	iT = NumColBond();
//	for (int k=0; k<iT; k++) BondArrayCollision[k].ResetBond();

	CurTime = (vfloat)0.0;
	CurStepCount = 0;

	ClearHistories();

	SS.Clear();
#endif
}

/*! Given the current state of the simulation (Voxel positions and velocities) and information about the current environment, advances the simulation by the maximum stable timestep. 
The integration scheme denoted by the CurIntegrator member variable is used.
Calculates some relevant system statistics such as maximum displacements and velocities and total force.
Returns true if the time step was successful, false otherwise.
@param[out] pRetMessage Pointer to an initialized string. Messages generated in this function will be appended to the string.
*/
bool CVX_Sim::TimeStep(std::string* pRetMessage)
{
#ifdef VX2
	if(IsFeatureEnabled(VXSFEAT_TEMPERATURE_VARY)) UpdateMatTemps(); //updates the temperatures
	
	if (IsFeatureEnabled(VXSFEAT_VOLUME_EFFECTS)) dt = DtFrac*Vx.recommendedTimeStep();
	else dt = DtFrac*OptimalDt;
	
	CurTime += dt; //keep track of time!
	CurStepCount++; //keep track of current step...

		//update information to calculate
	switch (GetStopConditionType()){ //may need to calculate certain items depending on stop condition
		case SC_CONST_MAXENERGY: StatToCalc |= CALCSTAT_KINE; StatToCalc |= CALCSTAT_STRAINE; break;
		case SC_MIN_KE: StatToCalc |= CALCSTAT_KINE; break;
		case SC_MIN_MAXMOVE: StatToCalc |= CALCSTAT_VEL; break;
	}
	if (IsFeatureEnabled(VXSFEAT_EQUILIBRIUM_MODE)) StatToCalc |= CALCSTAT_KINE;

	if (!Vx.doTimeStep(dt)){
		if (pRetMessage) *pRetMessage = "Simulation Diverged. Please reduce forces or accelerations.\n";	
		return false;
	}

	if (IsFeatureEnabled(VXSFEAT_EQUILIBRIUM_MODE) && KineticEDecreasing()){
		ZeroAllMotion(); MotionZeroed = true;} 
	else MotionZeroed = false;
	UpdateStats(pRetMessage);

	return true;
#else
	bool SelfColEnabled = IsFeatureEnabled(VXSFEAT_COLLISIONS);
	bool EquilibriumEnabled = IsFeatureEnabled(VXSFEAT_EQUILIBRIUM_MODE);

	if(SelfColEnabled){
		try {UpdateCollisions();} //update self intersection lists if necessary
		catch (std::bad_alloc&){if (pRetMessage) *pRetMessage += "Insufficient memory. Reduce model size."; return false;} //catch if we run out of memory
	}
	else if (!SelfColEnabled && ColEnableChanged){ColEnableChanged=false; DeleteCollisionBonds();}

	UpdateMatTemps(); //updates the temperatures

	//update information to calculate
	switch (GetStopConditionType()){ //may need to calculate certain items depending on stop condition
	case SC_CONST_MAXENERGY: StatToCalc |= CALCSTAT_KINE; StatToCalc |= CALCSTAT_STRAINE; break;
	case SC_MIN_KE: StatToCalc |= CALCSTAT_KINE; break;
	case SC_MIN_MAXMOVE: StatToCalc |= CALCSTAT_VEL; break;
	}
	if (EquilibriumEnabled) StatToCalc |= CALCSTAT_KINE;

	if (!Integrate()){
		if (pRetMessage) *pRetMessage = "Simulation Diverged. Please reduce forces or accelerations.\n";	
		return false;
	}
	
	if (EquilibriumEnabled && KineticEDecreasing()){ ZeroAllMotion(); MotionZeroed = true;} 
	else MotionZeroed = false;
	UpdateStats(pRetMessage);
	return true;
#endif
}

void CVX_Sim::EnableEquilibriumMode(bool Enabled)
{
//	EquilibriumModeEnabled = Enabled;
//	if (EquilibriumModeEnabled){
	if (Enabled){
		MemBondDampZ = BondDampingZ;
		MemSlowDampingZ = SlowDampingZ;
	//	MemMaxVelEnabled = IsFeatureEnabled(VXSFEAT_MAX_VELOCITY); // MaxVelLimitEnabled;

		for (int i=0; i<Vx.materialCount(); i++){ //set for each material
			CVX_Material* pMat = Vx.material(i);
			if (i==0){
				MemBondDampZ = pMat->internalDamping();
				MemSlowDampingZ = pMat->globalDamping();
			}
			pMat->setInternalDamping(1.0);
			pMat->setGlobalDamping(0.0);
		}

	//	BondDampingZ = 0.1;
	//	SlowDampingZ = 0;
	//	EnableFeature(VXSFEAT_MAX_VELOCITY, false);

//		MaxVelLimitEnabled = false;
	}
	else {
		for (int i=0; i<Vx.materialCount(); i++){ //set for each material
			CVX_Material* pMat = Vx.material(i);
			pMat->setInternalDamping(MemBondDampZ);
			pMat->setGlobalDamping(MemSlowDampingZ);
		}


//		BondDampingZ = MemBondDampZ;
//		SlowDampingZ = MemSlowDampingZ;
//		EnableFeature(VXSFEAT_MAX_VELOCITY, MemMaxVelEnabled);
//		MaxVelLimitEnabled = MemMaxVelEnabled;
	}
}

void CVX_Sim::EnableVolumeEffects(bool Enabled){
	if (Vx.materialCount() != muMemory.size())
		return;

	if (Enabled){
		for (int i=0; i<Vx.materialCount(); i++) Vx.material(i)->setPoissonsRatio(muMemory[i]);
	}
	else {
		for (int i=0; i<Vx.materialCount(); i++) Vx.material(i)->setPoissonsRatio(0);
	}
	
	OptimalDt = Vx.recommendedTimeStep();
}

bool CVX_Sim::IsVolumeEffectsEnabled()
{
	for (int i=0; i<Vx.materialCount(); i++) Vx.material(i)->setPoissonsRatio(muMemory[i]);

	return false;
}


void CVX_Sim::ZeroAllMotion(void)
{
	const std::vector<CVX_Voxel*>* pVoxList = Vx.voxelList();
	for (std::vector<CVX_Voxel*>::const_iterator it = pVoxList->begin(); it != pVoxList->end(); it++){
		(*it)->haltMotion();
	}

	//int NumVoxLoc = NumVox();
	//for (int i=0; i<NumVoxLoc; i++){
	//	VoxArray[i].ZeroMotion();
	//}
}


//void CVX_Sim::SetStopConditionType(StopCondition StopConditionTypeIn)
//{
//	//Type
//	if (StopConditionType != StopConditionTypeIn){
//		StopConditionType = StopConditionTypeIn;
//
//		if (StopCondRqsEnergy()){ //enable energy history at 100 (or more)
//			EnableEnergyHistory(501);
//		}
//		else { //disable energy history if eq mode doesn't need it
//			if (!EquilibriumModeEnabled) DisableEnergyHistory();
//		}
//	}
//}

bool CVX_Sim::StopConditionMet(void) //have we met the stop condition yet?
{
	if (CurStepCount<2*HISTORY_SIZE) return false;

	int numJump; //how many timesteps to look back in order to have 10 data points within the history length
	vfloat fNumVoxInv;
	if (StopConditionType==SC_CONST_MAXENERGY || StopConditionType==SC_MIN_KE || StopConditionType==SC_MIN_MAXMOVE){
		fNumVoxInv = 1.0/(float)Vx.voxelCount();
//		fNumVoxInv = 1.0/(float)NumVox();
		numJump = HISTORY_SIZE/10;
	}

	switch(StopConditionType){
		case SC_NONE: return false;
		case SC_MAX_TIME_STEPS: return (CurStepCount>(int)(StopConditionValue+0.5))?true:false;
		case SC_MAX_SIM_TIME: return CurTime>StopConditionValue?true:false;
		case SC_TEMP_CYCLES:  return CurTime>pEnv->GetTempPeriod()*StopConditionValue?true:false;
		case SC_CONST_MAXENERGY:{
			vfloat IniTotVal = TotEHistory[0];
			for (int i=numJump; i<HISTORY_SIZE; i+=numJump){
				if (TotEHistory[i] == -1) return false;
				if (abs(TotEHistory[i]-IniTotVal)*fNumVoxInv > 0.001*StopConditionValue) return false;
			}
			return true;
		  }
		case SC_MIN_KE:{
			for (int i=0; i<HISTORY_SIZE; i+=numJump){
				if (KinEHistory[i] == -1) return false;
				if (KinEHistory[i]*fNumVoxInv > 0.001*StopConditionValue) return false;
			}
			return true;
		  }
		case SC_MIN_MAXMOVE:{
			for (int i=0; i<HISTORY_SIZE; i+=numJump){
				if (MaxMoveHistory[i] == -1) return false;
				if (MaxMoveHistory[i] > 0.001*StopConditionValue) return false;
			}
			return true;
		}

		default: return false;
	}
}

bool CVX_Sim::UpdateStats(std::string* pRetMessage) //updates simulation state (SS)
{
	//if (SelfColEnabled) StatToCalc |= CALCSTAT_VEL; //always need velocities if self collisition is enabled
	if (IsFeatureEnabled(VXSFEAT_COLLISIONS)) StatToCalc |= CALCSTAT_VEL; //always need velocities if self collisition is enabled
	if (StatToCalc == CALCSTAT_NONE) return true;
	bool CCom=StatToCalc&CALCSTAT_COM, CDisp=StatToCalc&CALCSTAT_DISP, CVel=StatToCalc & CALCSTAT_VEL, CKinE=StatToCalc&CALCSTAT_KINE, CStrE=StatToCalc&CALCSTAT_STRAINE, CEStrn=StatToCalc&CALCSTAT_ENGSTRAIN, CEStrs=StatToCalc&CALCSTAT_ENGSTRESS, CPressure=StatToCalc&CALCSTAT_PRESSURE;

	if (CCom) SS.CurCM = GetCM(); //calculate center of mass

	//update the overall statisics (can't do this within threaded loops and be safe without mutexes...
	vfloat tmpMaxVoxDisp2 = 0, tmpMaxVoxVel2 = 0, tmpMaxVoxKineticE = 0, tmpMaxVoxStrainE = 0, tmpMaxPressure = -FLT_MAX, tmpMinPressure = FLT_MAX;
	vfloat tmpMaxBondStrain=0, tmpMaxBondStress=0, tmpTotalObjKineticE = 0, tmpTotalObjStrainE=0;
	Vec3D<> tmpTotalObjDisp(0,0,0);

	if (CDisp || CVel || CKinE || CPressure){
		int nVox = Vx.voxelCount();// NumVox();
		for (int i=0; i<nVox; i++){ //for each voxel
//			if (i == InputVoxSInd) continue;
//			const CVXS_Voxel* it = &VoxArray[i]; //pointer to this voxel
			const CVX_Voxel* it = Vx.voxel(i); //pointer to this voxel

			if (CDisp) { //Displacements
//				tmpTotalObjDisp += it->GetCurVel().Abs()*dt; //keep track of displacements on global object
				tmpTotalObjDisp += it->velocity().Abs()*dt; //keep track of displacements on global object
//				const vfloat ThisMaxVoxDisp2 = (it->GetCurPos()-it->GetNominalPosition()).Length2();
				const float ThisMaxVoxDisp2 = it->displacement().Length2();
				if (ThisMaxVoxDisp2 > tmpMaxVoxDisp2) tmpMaxVoxDisp2 = ThisMaxVoxDisp2;
			}

			if (CVel) { //Velocities
				const vfloat ThisMaxVoxVel2 = it->velocity().Length2(); //it->GetCurVel().Length2();
				if (ThisMaxVoxVel2 > tmpMaxVoxVel2) tmpMaxVoxVel2 = ThisMaxVoxVel2;
			}
			if (CKinE) { // kinetic energy
				const vfloat ThisMaxKineticE = it->kineticEnergy(); // kineticEnergy(); // it->GetCurKineticE();
				if (ThisMaxKineticE > tmpMaxVoxKineticE) tmpMaxVoxKineticE = ThisMaxKineticE;
				tmpTotalObjKineticE += ThisMaxKineticE; //keep track of total kinetic energy
			}
			if (CPressure){
				const vfloat ThisPressure = it->pressure();
				if (ThisPressure > tmpMaxPressure) tmpMaxPressure = ThisPressure;
				if (ThisPressure < tmpMinPressure) tmpMinPressure = ThisPressure;

			}
		}

		if (CDisp){ //Update SimState (SS)
			tmpTotalObjDisp /= nVox;
			SS.TotalObjDisp = tmpTotalObjDisp;
			SS.NormObjDisp = tmpTotalObjDisp.Length();
			SS.MaxVoxDisp = sqrt(tmpMaxVoxDisp2);
		}

		if (CVel) SS.MaxVoxVel = sqrt(tmpMaxVoxVel2);
		if (CKinE) {
			SS.MaxVoxKinE = tmpMaxVoxKineticE;
			SS.TotalObjKineticE = tmpTotalObjKineticE;
		}
		if (CPressure){
			SS.MaxPressure = tmpMaxPressure;
			SS.MinPressure = tmpMinPressure;
		}
	}

	if (CStrE || CEStrn || CEStrs){
		int nLink = Vx.linkCount();
		for (int i=0; i<nLink; i++){ //for each voxel
			CVX_Link* it = Vx.link(i);
	//	for (std::vector<CVXS_BondInternal>::iterator it = BondArrayInternal.begin(); it != BondArrayInternal.end(); it++){
			if (CStrE){
				const vfloat ThisMaxStrainE =  it->strainEnergy(); // GetStrainEnergy();
				if (ThisMaxStrainE > tmpMaxVoxStrainE) tmpMaxVoxStrainE = ThisMaxStrainE;
				tmpTotalObjStrainE += ThisMaxStrainE;
			}

			if (CEStrn && it->axialStrain() > tmpMaxBondStrain) tmpMaxBondStrain = it->axialStrain(); //shouldn't these pull from bonds? would make more sense...
			if (CEStrs && it->axialStress() > tmpMaxBondStress) tmpMaxBondStress = it->axialStress();

//			if (CEStrn && it->GetEngStrain() > tmpMaxBondStrain) tmpMaxBondStrain = it->GetEngStrain(); //shouldn't these pull from bonds? would make more sense...
//			if (CEStrs && it->GetEngStress() > tmpMaxBondStress) tmpMaxBondStress = it->GetEngStress();
		}
	
		//Updata SimState (SS)
		if (CStrE){
			SS.MaxBondStrainE = tmpMaxVoxStrainE;
			SS.TotalObjStrainE = tmpTotalObjStrainE;
		}

		if (CEStrn) SS.MaxBondStrain = tmpMaxBondStrain;
		if (CEStrs) SS.MaxBondStress = tmpMaxBondStress;

	}


	//update histories
	MaxMoveHistory.push_front(CVel ? SS.MaxVoxVel*dt : -1.0); MaxMoveHistory.pop_back();
	KinEHistory.push_front(CKinE ? SS.TotalObjKineticE : -1.0); KinEHistory.pop_back();
	TotEHistory.push_front((CStrE && CKinE) ? SS.TotalObjKineticE + SS.TotalObjStrainE : -1.0); TotEHistory.pop_back();

	return true;
}

//
//vfloat CVX_Sim::CalcMaxDt(void)
//{
//	vfloat MaxFreq2 = 0; //maximum frequency in the simulation
////	vfloat MaxRFreq2 = 0; //maximum frequency in the simulation
//
//	bool VolEffectEnabled = IsFeatureEnabled(VXSFEAT_VOLUME_EFFECTS);
//	int iT = NumBond();
//	if (iT != 0){
//		for (int i=0; i<iT; i++){
//	//		if (i==InputBondInd) continue; //zero mass of input voxel causes problems
//			if (VolEffectEnabled){
//				if (BondArrayInternal[i].GetEffectiveStiffness()/BondArrayInternal[i].GetpV1()->GetMass() > MaxFreq2) MaxFreq2 = BondArrayInternal[i].GetEffectiveStiffness()/BondArrayInternal[i].GetpV1()->GetMass();
//				if (BondArrayInternal[i].GetEffectiveStiffness()/BondArrayInternal[i].GetpV2()->GetMass() > MaxFreq2) MaxFreq2 = BondArrayInternal[i].GetEffectiveStiffness()/BondArrayInternal[i].GetpV2()->GetMass();
//			}
//			else {
//				if (BondArrayInternal[i].GetLinearStiffness()/BondArrayInternal[i].GetpV1()->GetMass() > MaxFreq2) MaxFreq2 = BondArrayInternal[i].GetLinearStiffness()/BondArrayInternal[i].GetpV1()->GetMass();
//				if (BondArrayInternal[i].GetLinearStiffness()/BondArrayInternal[i].GetpV2()->GetMass() > MaxFreq2) MaxFreq2 = BondArrayInternal[i].GetLinearStiffness()/BondArrayInternal[i].GetpV2()->GetMass();
//			}
//		}
//	}
//	else { //special (unlikely) case with no bonds, but (potentially) voxels.
//		int VoxCount = NumVox();
//		if (VoxCount == 0) return 0;
//		else {
//			for (int i=0; i<VoxCount; i++){
//				if (VoxArray[i].GetEMod()/VoxArray[i].GetMass() > MaxFreq2) MaxFreq2 = VoxArray[i].GetEMod()/VoxArray[i].GetMass();
//			}
//		}
//	}
//
//	//calculate dt: (as large as possible...)
//	vfloat MaxFreq = sqrt(MaxFreq2);
//	return 1.0/(MaxFreq*2*(vfloat)3.1415926); //convert to time... (seconds)
//
//}
//
//void CVX_Sim::UpdateCollisions(void) // Called every timestep to watch for collisions
//{
//	//self intersection accumulator
//	MaxDispSinceLastBondUpdate += abs(SS.MaxVoxVel*dt/LocalVXC.GetLatticeDim());
//
//	if (CurColSystem == COL_BASIC_HORIZON || CurColSystem == COL_SURFACE_HORIZON){
//		if (MaxDispSinceLastBondUpdate > (CollisionHorizon-1.0)/2 || ColEnableChanged){ //if we want to check for self intersection (slow!)
//			ColEnableChanged = false;
//			CalcL1Bonds(CollisionHorizon); //doesn't need to be done every time...
//			MaxDispSinceLastBondUpdate = 0.0;
//		}
//	}
//	else { //if COL_BASIC || COL_SURFACE
//		CalcL1Bonds(CollisionHorizon);
//	}
//}

void CVX_Sim::UpdateMatTemps(void) //updates expansions for each material
{
//#ifdef VX2
	for (int iz=Vx.indexMinZ(); iz<=Vx.indexMaxZ(); iz++){
		for (int iy=Vx.indexMinY(); iy<=Vx.indexMaxY(); iy++){
			for (int ix=Vx.indexMinX(); ix<=Vx.indexMaxX(); ix++){
				CVX_Voxel* pV = Vx.voxel(ix, iy, iz);
				float thisTemp = 0;


				if (pV != NULL){
					if (IsFeatureEnabled(VXSFEAT_TEMPERATURE)) pV->setTemperature(pEnv->UpdateCurTemp(CurTime, &LocalVXC)-pEnv->GetTempBase()); //pEnv->GetTempAmplitude());
					else pV->setTemperature(0);
				}
			}
		}
	}

//#else
////	pEnv->UpdateCurTemp(CurTime, &LocalVXC);
//#endif
}

void CVX_Sim::UpdateMuMemory(void) //updates array for turning poissons ratio on and off.
{
	bool wasVolEnabled = IsFeatureEnabled(VXSFEAT_VOLUME_EFFECTS);
	if (!wasVolEnabled) EnableFeature(VXSFEAT_VOLUME_EFFECTS);

	muMemory.clear();
	for (int i=0; i<Vx.materialCount(); i++){ //for each material
		muMemory.push_back(Vx.material(i)->poissonsRatio()); //remember for toggleing volume effects on and off.
	}

	if (!wasVolEnabled) EnableFeature(VXSFEAT_VOLUME_EFFECTS, false);
}


//
//bool CVX_Sim::Integrate()
//{
//	//Euler integration:
//
//	//Update Forces...
//	int iT = NumBond();
////	BondInput->UpdateBond();
//
//	bool Diverged = false;
////#pragma omp parallel for
//	for (int i=0; i<iT; i++){
//		BondArrayInternal[i].UpdateBond();
//		if (BondArrayInternal[i].GetEngStrain() > 100) Diverged = true; //catch divergent condition! (if any thread sets true we will fail, so don't need mutex...
//	}
//	if (Diverged) return false;
//
////	Vec3D<> F1a = BondArrayInternal[0].GetForce1();
////	Vec3D<> F1b = BondArrayInternal[2].GetForce1();
////	Vec3D<> M1a = BondArrayInternal[0].GetMoment1();
////	Vec3D<> M1b = BondArrayInternal[2].GetMoment1();
//
//	iT = NumColBond();
////#pragma omp parallel for
//	for (int i=0; i<iT; i++){
//		BondArrayCollision[i].UpdateBond();
//	}
//
//
//	//if (!DtFrozen){ //for now, dt cannot change within the simulation (and this is a cycle hog)
//	if (IsFeatureEnabled(VXSFEAT_VOLUME_EFFECTS)) OptimalDt = CalcMaxDt(); //calculate every time for now when volume effects are enabled
//	dt = DtFrac*OptimalDt;
//	//}
//
//	//Update positions... need to do this seperately if we're going to do damping in the summing forces stage.
//	iT = NumVox();
//
////#pragma omp parallel for
//	for (int i=0; i<iT; i++) { VoxArray[i].EulerStep();}
//
//	//End Euler integration
//
//	
////	Vec3D<> Pa = VoxArray[0].GetCurPos();
////	Vec3D<> Pb = VoxArray[3].GetCurPos();
////	Vec3D<> Aa = VoxArray[0].GetCurAngle();
////	Vec3D<> Ab = VoxArray[3].GetCurAngle();
//
//
//	CurTime += dt; //keep track of time!
//	CurStepCount++; //keep track of current step...
//
//	return true;
//}
//
//#ifdef USE_OPEN_GL
//
//void CVX_Sim::Draw(int Selected, bool ViewSection, int SectionLayer)
//{
//	if (!Initalized) return;
//
//	if (CurViewMode == RVM_NONE) return;
//	else if (CurViewMode == RVM_VOXELS){ 
//		switch (CurViewVox){
//		case RVV_DISCRETE: DrawGeometry(Selected, ViewSection, SectionLayer); break; //section view only currently enabled in voxel view mode
//		case RVV_DEFORMED: DrawVoxMesh(Selected); break;
//		case RVV_SMOOTH: DrawSurfMesh(); break;
//		}
//	}
//	else { //CurViewMode == RVT_BONDS
//		DrawBonds();
//		DrawStaticFric();
//	}
//	if (ViewAngles)	DrawAngles();
//	if (ViewForce) DrawForce();
//	if (IsFeatureEnabled(VXSFEAT_FLOOR)) DrawFloor(); //draw the floor if its in use
////	if (pEnv->IsFloorEnabled()) DrawFloor(); //draw the floor if its in use
//
//	NeedStatsUpdate=true;
//}
//
//void CVX_Sim::DrawForce(void)
//{
//	//TODO
//}
//
//void CVX_Sim::DrawFloor(void)
//{
//
//	//TODO: build an openGL list 
//	vfloat Size = LocalVXC.GetLatticeDim()*4;
//	vfloat sX = 1.5*Size;
//	vfloat sY = .866*Size;
//
//	glEnable(GL_LIGHTING);
//
//	glLoadName (-1); //never want to pick floor
//
//	glNormal3d(0.0, 0.0, 1.0);
//	for (int i=-20; i <=30; i++){
//		for (int j=-40; j <=60; j++){
//			glColor4d(0.6, 0.7+0.2*((int)(1000*sin((float)(i+110)*(j+106)*(j+302)))%10)/10.0, 0.6, 1.0);
//			glBegin(GL_TRIANGLE_FAN);
//			glVertex3d(i*sX, j*sY, 0.0);
//			glVertex3d(i*sX+0.5*Size, j*sY, 0.0);
//			glVertex3d(i*sX+0.25*Size, j*sY+0.433*Size, 0.0);
//			glVertex3d(i*sX-0.25*Size, j*sY+0.433*Size, 0.0);
//			glVertex3d(i*sX-0.5*Size, j*sY, 0.0);
//			glVertex3d(i*sX-0.25*Size, j*sY-0.433*Size, 0.0);
//			glVertex3d(i*sX+0.25*Size, j*sY-0.433*Size, 0.0);
//			glVertex3d(i*sX+0.5*Size, j*sY, 0.0);
//			glEnd();
//
//			glColor4d(0.6, 0.7+0.2*((int)(1000*sin((float)(i+100)*(j+103)*(j+369)))%10)/10.0, 0.6, 1.0);
//			
//			glBegin(GL_TRIANGLE_FAN);
//			glVertex3d(i*sX+.75*Size, j*sY+0.433*Size, 0.0);
//			glVertex3d(i*sX+1.25*Size, j*sY+0.433*Size, 0.0);
//			glVertex3d(i*sX+Size, j*sY+0.866*Size, 0.0);
//			glVertex3d(i*sX+0.5*Size, j*sY+0.866*Size, 0.0);
//			glVertex3d(i*sX+0.25*Size, j*sY+0.433*Size, 0.0);
//			glVertex3d(i*sX+0.5*Size, j*sY, 0.0);
//			glVertex3d(i*sX+Size, j*sY, 0.0);
//			glVertex3d(i*sX+1.25*Size, j*sY+0.433*Size, 0.0);
//			glEnd();
//		}
//	}
//}
//
//void CVX_Sim::DrawGeometry(int Selected, bool ViewSection, int SectionLayer)
//{
////	bool DrawInputVoxel = true;
//	Vec3D<> Center;
//	Vec3D<> tmp(0,0,0);
//
//	int iT = NumVox();
//	int x, y, z;
//	CColor ThisColor;
//	for (int i = 0; i<iT; i++) //go through all the voxels...
//	{
//		pEnv->pObj->GetXYZNom(&x, &y, &z, StoXIndexMap[i]);
//		if (ViewSection && z>SectionLayer) continue; //exit if obscured in a section view!
//
//
//		Center = VoxArray[i].GetCurPos();
//
//		ThisColor = GetCurVoxColor(i, Selected);
//		glColor4d(ThisColor.r, ThisColor.g, ThisColor.b, ThisColor.a);
//		
//
//		glPushMatrix();
//		glTranslated(Center.x, Center.y, Center.z);
//
//		glLoadName (StoXIndexMap[i]); //to enable picking
//
//		//generate rotation matrix here!!! (from quaternion)
//		Vec3D<> Axis;
//		vfloat AngleAmt;
//		Quat3D<>(VoxArray[i].GetCurAngle()).AngleAxis(AngleAmt, Axis);
//		glRotated(AngleAmt*180/3.1415926, Axis.x, Axis.y, Axis.z);
//	
//		Vec3D<> CurrentSizeDisplay = VoxArray[i].GetSizeCurrent();
//		glScaled(CurrentSizeDisplay.x, CurrentSizeDisplay.y, CurrentSizeDisplay.z); 
//
//		LocalVXC.Voxel.DrawVoxel(&tmp, 1); //draw unit size since we scaled just now
//		
//		glPopMatrix();
//	}
//
//	//if (DrawInputVoxel){
//	//	Vec3D<> tmp(0,0,0);
//	//	Center = GetInputVoxel()->GetCurPos();
//	//	glColor4d(1.0, 0.2, 0.2, 1.0);
//	//	glPushMatrix();
//	//	glTranslated(Center.x, Center.y, Center.z);	
//	//	vfloat Scale = LocalVXC.GetLatDimEnv().x; //todo: enforce cubic voxels only
//	//	glScaled(Scale, Scale, Scale);
//
//	//	LocalVXC.Voxel.DrawVoxel(&tmp, 1); //LocalVXC.GetLatticeDim()); //[i].CurSize.x); //, LocalVXC.Lattice.Z_Dim_Adj);
//	//	
//	//	glPopMatrix();
//	//}
//}
//
//CColor CVX_Sim::GetCurVoxColor(int SIndex, int Selected)
//{
//	if (StoXIndexMap[SIndex] == Selected) return CColor(1.0f, 0.0f, 1.0f, 1.0f); //highlight selected voxel (takes precedence...)
//
//	switch (CurViewCol) {
//		case RVC_TYPE:
//			float R, G, B, A;
////			LocalVXC.GetLeafMat(VoxArray[SIndex].GetVxcIndex())->GetColorf(&R, &G, &B, &A);
//			VoxArray[SIndex].GetpMaterial()->GetColorf(&R, &G, &B, &A);
//			return CColor(R, G, B, A);
//			break;
//		case RVC_KINETIC_EN:
//			if (SS.MaxVoxKinE == 0) return GetJet(0);
//			return GetJet(VoxArray[SIndex].GetCurKineticE() / SS.MaxVoxKinE);
//			break;
//		case RVC_DISP:
//			if (SS.MaxVoxDisp == 0) return GetJet(0);
//			return GetJet(VoxArray[SIndex].GetCurAbsDisp() / SS.MaxVoxDisp);
//			break;
//		case RVC_STATE:
//			if (VoxArray[SIndex].GetBroken()) return CColor(1.0f, 0.0f, 0.0f, 1.0f);
//			else if (VoxArray[SIndex].GetYielded()) return CColor(1.0f, 1.0f, 0.0f, 1.0f);
//			else return CColor(1.0f, 1.0f, 1.0f, 1.0f);
//			break;
//		case RVC_STRAIN_EN:
//			if (SS.MaxBondStrainE == 0) return GetJet(0);
//			return GetJet(VoxArray[SIndex].GetMaxBondStrainE() / SS.MaxBondStrainE);
//			break;
//		case RVC_STRAIN:
//			if (SS.MaxBondStrain == 0) return GetJet(0);
//			return GetJet(VoxArray[SIndex].GetMaxBondStrain() / SS.MaxBondStrain);
//			break;
//		case RVC_STRESS:
//			if (SS.MaxBondStress == 0) return GetJet(0);
//			return GetJet(VoxArray[SIndex].GetMaxBondStress() / SS.MaxBondStress);
//			break;
//
//	
//		default:
//			return CColor(1.0f,1.0f,1.0f, 1.0f);
//			break;
//	}
//}
//
//CColor CVX_Sim::GetInternalBondColor(CVXS_BondInternal* pBond)
//{
//	switch (CurViewCol) {
//		case RVC_TYPE:
//			if (pBond->IsSmallAngle()) return CColor(0.3, 0.7, 0.3, 1.0);
//			else return CColor(0.0, 0.0, 0.0, 1.0);
//			break;
//		case RVC_KINETIC_EN:
//			if (SS.MaxVoxKinE == 0) return GetJet(0);
//			return GetJet(pBond->GetMaxVoxKinE() / SS.MaxVoxKinE);
//			break;
//		case RVC_DISP:
//			if (SS.MaxVoxDisp == 0) return GetJet(0);
//			return GetJet(pBond->GetMaxVoxDisp() / SS.MaxVoxDisp);
//			break;
//		case RVC_STATE:
//			if (pBond->IsBroken()) return CColor(1.0f, 0.0f, 0.0f, 1.0f);
//			else if (pBond->IsYielded()) return CColor(1.0f, 1.0f, 0.0f, 1.0f);
//			else return CColor(1.0f, 1.0f, 1.0f, 1.0f);
//			break;
//		case RVC_STRAIN_EN:
//			if (SS.MaxBondStrainE == 0) return GetJet(0);
//			return GetJet(pBond->GetStrainEnergy() / SS.MaxBondStrainE);
//			break;
//		case RVC_STRAIN:
//			if (SS.MaxBondStrain == 0) return GetJet(0);
//			return GetJet(pBond->GetEngStrain() / SS.MaxBondStrain);
//			break;
//		case RVC_STRESS:
//			if (SS.MaxBondStress == 0) return GetJet(0);
//			return GetJet(pBond->GetEngStress() / SS.MaxBondStress);
//			break;
//	
//		default:
//			return CColor(0.0f,0.0f,0.0f,1.0f);
//			break;
//	}
//}
//
//CColor CVX_Sim::GetCollisionBondColor(CVXS_BondCollision* pBond)
//{
//	if (!IsFeatureEnabled(VXSFEAT_COLLISIONS)) return CColor(0.0, 0.0, 0.0, 0.0); //Hide me
//	vfloat Force = pBond->GetForce1().Length(); //check which force to use!
//	if (Force == 0.0) return CColor(0.3, 0.3,1.0, 1.0);
//	else return CColor(1.0, 0.0, 0.0, 1.0);
//}
//
//
////CColor CVX_Sim::GetCurBondColor(CVXS_Bond* pBond)
////{
////	switch (pBond->GetBondType()){
////		case B_LINEAR:
////
////			break;
////		case B_LINEAR_CONTACT: {
////
////		break;
////
////		case B_INPUT_LINEAR_NOROT:
//////			if (!Dragging) return CColor(0.0, 0.0, 0.0, 0.0); //Hide me
////			return CColor(1.0, 0.0, 0.0, 1.0);
////		break;
////		default:
////			return CColor(0.0, 0.0, 0.0, 0.0); //Hide me
////			break;
////
////	}
////}
//
//void CVX_Sim::DrawSurfMesh(int Selected)
//{
//	SurfMesh.UpdateMesh(Selected); //updates the generated mesh
//	SurfMesh.Draw();
//}
//
//void CVX_Sim::DrawVoxMesh(int Selected)
//{
//	VoxMesh.UpdateMesh(Selected); //updates the generated mesh
//	VoxMesh.Draw();
//}
//
//
//void CVX_Sim::DrawBonds(void)
//{
////	bool DrawInputBond = true;
//
//	Vec3D<> P1, P2;
//	CVXS_Voxel* pV1, *pV2;
//
//	float PrevLineWidth;
//	glGetFloatv(GL_LINE_WIDTH, &PrevLineWidth);
//	glLineWidth(3.0);
//	glDisable(GL_LIGHTING);
//
//	int iT = NumBond();
//	glBegin(GL_LINES);
//	glLoadName (-1); //to disable picking
//	for (int i = 0; i<iT; i++) //go through all the bonds...
//	{
//		pV1 = BondArrayInternal[i].GetpV1(); pV2 = BondArrayInternal[i].GetpV2();
//
//		CColor ThisColor = GetInternalBondColor(&BondArrayInternal[i]);
//		P1 = pV1->GetCurPos();
//		P2 = pV2->GetCurPos();
//
//		glColor4f(ThisColor.r, ThisColor.g, ThisColor.b, ThisColor.a);
//		//TODO:sweet curved bonds!
//		//if (CurViewVox == RVV_SMOOTH){
//		//	Quat3D A1 = pV1->GetCurAngle();
//		//	Quat3D A2 = pV1->GetCurAngle();
//		//}
//		//else {
//			if (ThisColor.a != 0.0) {glVertex3f((float)P1.x, (float)P1.y, (float)P1.z); glVertex3f((float)P2.x, (float)P2.y, (float)P2.z);}
////		}
//	}
//
//	iT = NumColBond();
//	glBegin(GL_LINES);
//	glLoadName (-1); //to disable picking
//	for (int i = 0; i<iT; i++) //go through all the bonds...
//	{
//		pV1 = BondArrayCollision[i].GetpV1(); pV2 = BondArrayCollision[i].GetpV2();
//
//		CColor ThisColor = GetCollisionBondColor(&BondArrayCollision[i]);
//		P1 = pV1->GetCurPos();
//		P2 = pV2->GetCurPos();
//
//		glColor4f(ThisColor.r, ThisColor.g, ThisColor.b, ThisColor.a);
//			if (ThisColor.a != 0.0) {glVertex3f((float)P1.x, (float)P1.y, (float)P1.z); glVertex3f((float)P2.x, (float)P2.y, (float)P2.z);}
//	}
//
//
//	////input bond
//	//if (DrawInputBond && BondInput->GetpV1() && BondInput->GetpV2()){
//	//	glColor4f(1.0, 0, 0, 1.0);
//	//	P1 = BondInput->GetpV1()->GetCurPos();
//	//	P2 = BondInput->GetpV2()->GetCurPos();
//	//	glVertex3f((float)P1.x, (float)P1.y, (float)P1.z); glVertex3f((float)P2.x, (float)P2.y, (float)P2.z);
//	//}
//
//	glEnd();
//
//
//	Vec3D<> Center;
//	iT = NumVox();
//	glPointSize(5.0);
//	Vec3D<> tmp(0,0,0);
//	for (int i = 0; i<iT; i++) //go through all the voxels...
//	{
//		//mostly copied from Voxel drawing function!
//		Center = VoxArray[i].GetCurPos();
//		glColor4d(0.2, 0.2, 0.2, 1.0);
//	//	glLoadName (StoXIndexMap[i]); //to enable picking
//
//		glPushMatrix();
//		glTranslated(Center.x, Center.y, Center.z);
//		glLoadName (StoXIndexMap[i]); //to enable picking
//
//		//generate rotation matrix here!!! (from quaternion)
//		Vec3D<> Axis;
//		vfloat AngleAmt;
//		Quat3D<>(VoxArray[i].GetCurAngle()).AngleAxis(AngleAmt, Axis);
//		glRotated(AngleAmt*180/3.1415926, Axis.x, Axis.y, Axis.z);
//	
//		vfloat Scale = VoxArray[i].GetCurScale(); //show deformed voxel size
//		glScaled(Scale, Scale, Scale);
//
//		//LocalVXC.Voxel.DrawVoxel(&tmp, LocalVXC.Lattice.Lattice_Dim*(1+0.5*CurTemp * pMaterials[CVoxelArray[i].MatIndex].CTE), LocalVXC.Lattice.Z_Dim_Adj);
//		LocalVXC.Voxel.DrawVoxel(&tmp, 0.2); //LocalVXC.GetLatticeDim()); //[i].CurSize.x); //, LocalVXC.Lattice.Z_Dim_Adj);
//		
//		glPopMatrix();
//	}
//
//
//	glLineWidth(PrevLineWidth);
//	glEnable(GL_LIGHTING);
//
//}
//
//void CVX_Sim::DrawAngles(void)
//{
//	//draw directions
//	float PrevLineWidth;
//	glGetFloatv(GL_LINE_WIDTH, &PrevLineWidth);
//	glLineWidth(2.0);
//	glDisable(GL_LIGHTING);
//
//	glBegin(GL_LINES);
//
//	for (int i = 0; i<NumVox(); i++){ //go through all the voxels... (GOOD FOR ONLY SMALL DISPLACEMENTS, I THINK... think through transformations here!)
//		glColor3f(1,0,0); //+X direction
//		glVertex3d(VoxArray[i].GetCurPos().x, VoxArray[i].GetCurPos().y, VoxArray[i].GetCurPos().z);
//		Vec3D<> Axis1(LocalVXC.GetLatticeDim()/4,0,0);
//		Vec3D<> RotAxis1 = (VoxArray[i].GetCurAngle()*Quat3D<>(Axis1)*VoxArray[i].GetCurAngle().Conjugate()).ToVec();
//		glVertex3d(VoxArray[i].GetCurPos().x + RotAxis1.x, VoxArray[i].GetCurPos().y + RotAxis1.y, VoxArray[i].GetCurPos().z + RotAxis1.z);
//
//		glColor3f(0,1,0); //+Y direction
//		glVertex3d(VoxArray[i].GetCurPos().x, VoxArray[i].GetCurPos().y, VoxArray[i].GetCurPos().z);
//		Axis1 = Vec3D<>(0, LocalVXC.GetLatticeDim()/4,0);
//		RotAxis1 = (VoxArray[i].GetCurAngle()*Quat3D<>(Axis1)*VoxArray[i].GetCurAngle().Conjugate()).ToVec();
//		glVertex3d(VoxArray[i].GetCurPos().x + RotAxis1.x, VoxArray[i].GetCurPos().y + RotAxis1.y, VoxArray[i].GetCurPos().z + RotAxis1.z);
//
//		glColor3f(0,0,1); //+Z direction
//		glVertex3d(VoxArray[i].GetCurPos().x, VoxArray[i].GetCurPos().y, VoxArray[i].GetCurPos().z);
//		Axis1 = Vec3D<>(0,0, LocalVXC.GetLatticeDim()/4);
//		RotAxis1 = (VoxArray[i].GetCurAngle()*Quat3D<>(Axis1)*VoxArray[i].GetCurAngle().Conjugate()).ToVec();
//		glVertex3d(VoxArray[i].GetCurPos().x + RotAxis1.x, VoxArray[i].GetCurPos().y + RotAxis1.y, VoxArray[i].GetCurPos().z + RotAxis1.z);
//
//	}
//	glEnd();
//
//	glLineWidth(PrevLineWidth);
//	glEnable(GL_LIGHTING);
//}
//
//void CVX_Sim::DrawStaticFric(void)
//{
//	//draw triangle for points that are stuck via static friction
//	glBegin(GL_TRIANGLES);
//	glColor4f(255, 255, 0, 1.0);
//	vfloat dist = VoxArray[0].GetNominalSize()/3; //needs work!!
//	int iT = NumVox();
//	Vec3D<> P1;
//	for (int i = 0; i<iT; i++){ //go through all the voxels...
//		if (VoxArray[i].GetCurStaticFric()){ //draw point if static friction...
//			P1 = VoxArray[i].GetCurPos();
//			glVertex3f((float)P1.x, (float)P1.y, (float)P1.z); 
//			glVertex3f((float)P1.x, (float)(P1.y - dist/2), (float)(P1.z + dist));
//			glVertex3f((float)P1.x, (float)(P1.y + dist/2), (float)(P1.z + dist));
//		}
//	}
//	glEnd();
//}
//
//int CVX_Sim::StatRqdToDraw() //returns the stats bitfield that we need to calculate to draw the current view.
//{
//	if (CurViewMode == RVM_NONE) return CALCSTAT_NONE;
//	switch (CurViewCol){
//	case RVC_KINETIC_EN: return CALCSTAT_KINE; break;
//	case RVC_DISP: return CALCSTAT_DISP; break;
//	case RVC_STRAIN_EN: return CALCSTAT_STRAINE; break;
//	case RVC_STRAIN: return CALCSTAT_ENGSTRAIN; break;
//	case RVC_STRESS: return CALCSTAT_ENGSTRESS; break;
//	default: return CALCSTAT_NONE;
//	}
//}
//
//#endif //OPENGL
//
//void CVX_Sim::CalcL1Bonds(vfloat Dist) //creates contact bonds for all voxels within specified distance
//{
////	Dist = Dist*LocalVXC.GetLatticeDim();
//	vfloat Dist2;
//	vfloat FilterDist = Dist*1.5*pEnv->pObj->GetLatticeDim();//the distance to immediately discard voxels at. Calculations involving this number will not include local scaling of the voxel.
//	vfloat FilterDist2 = FilterDist*FilterDist;
//
//	//todo: only recalc for fast moving?
//	//todo: keep only nearby list of surface voxels
//
//	DeleteCollisionBonds();
//
//	if (CurColSystem == COL_SURFACE || COL_SURFACE_HORIZON){
//		//clear bond array past the m_NumBonds position
//		int SurfVoxCount = NumSurfVoxels();
//		for (int i=0; i<SurfVoxCount; i++){ //go through each combination of surface voxels...
//			int SIndex1 = SurfVoxels[i];
//			CVXS_Voxel* pV1 = &VoxArray[SIndex1]; //could cache the pointers...
//
//			for (int j=i+1; j<SurfVoxCount; j++){
//				int SIndex2 = SurfVoxels[j];
//				CVXS_Voxel* pV2 = &VoxArray[SIndex2]; //could cache the pointers...
//
//				Dist2 = (pV1->GetCurPos() - pV2->GetCurPos()).Length2();
//				if (Dist2 < FilterDist2 && !pV1->IsNearbyVox(SIndex2)){ //quick filter...
//					vfloat ActDist = Dist*(pV1->GetCurScale() + pV1->GetCurScale())*0.5; //ASSUMES ISOTROPIC!!
//
//					if (Dist2 < ActDist*ActDist) CreateColBond(SIndex1, SIndex2); //if within the threshold create temporary bond...
//				}
//			}
//		}
//	}
//	else { //check against all!
//		for (int i=0; i<NumVox(); i++){ //go through each combination of voxels...
//			int SIndex1 = i;
//
//			for (int j=i+1; j<NumVox(); j++){
//				int SIndex2 = j;
//				if (!VoxArray[SIndex1].IsNearbyVox(SIndex2)){
//
//					vfloat ActDist = Dist*(VoxArray[SIndex1].GetCurScale() + VoxArray[SIndex1].GetCurScale())*0.5; //ASSUMES ISOTROPIC!!
//					Dist2 = (VoxArray[SIndex1].GetCurPos() - VoxArray[SIndex2].GetCurPos()).Length2();
//					if (Dist2 < ActDist*ActDist){ //if within the threshold
//						CreateColBond(SIndex1, SIndex2); //create temporary bond...	
//					}
//
//				}
//			}
//		}
//	}
//
//
//	//todo: check for reallocation before doing this?
//	for (std::vector<CVXS_Voxel>::iterator it = VoxArray.begin(); it != VoxArray.end(); it++) it->UpdateColBondPointers();
//	
//	//UpdateAllBondPointers();
//}

Vec3D<> CVX_Sim::GetCM(void)
{
	vfloat TotalMass = 0;
	Vec3D<> Sum(0,0,0);
	int nVox = Vx.voxelCount(); 
	for (int i=0; i<nVox; i++){
		CVX_Voxel* it = Vx.voxel(i);
		vfloat ThisMass = it->material()->mass();
		Sum += it->position()*ThisMass;
		TotalMass += ThisMass;
	}

	return Sum/TotalMass;
//
//	vfloat TotalMass = 0;
//	Vec3D<> Sum(0,0,0);
//	int nVox = NumVox();
//	for (int i=0; i<nVox; i++){
////		if (i==InputVoxSInd) continue;
//		CVXS_Voxel* it = &VoxArray[i];
//		vfloat ThisMass = it->GetMass();
//		Sum += it->GetCurPos()*ThisMass;
//		TotalMass += ThisMass;
//	}
//
//	return Sum/TotalMass;
}

int CVX_Sim::GetNumTouchingFloor()
{
	int NumTouching = 0;

	int LocNumVox = Vx.voxelCount(); 
	for (int i=0; i<LocNumVox; i++){
		if (Vx.voxel(i)->floorPenetration() > 0) NumTouching++;
	}
	return NumTouching;

	//int NumTouching = 0;
	//int LocNumVox = NumVox();
	//for (int i=0; i<LocNumVox; i++){
	//	if (VoxArray[i].GetCurGroundPenetration() > 0) NumTouching++;
	//}
	//return NumTouching;
}

bool CVX_Sim::KineticEDecreasing(void)
{
	 if (KinEHistory[0]+KinEHistory[1]+KinEHistory[2] < KinEHistory[3]+KinEHistory[4]+KinEHistory[5] && !(KinEHistory[0] == 0 || KinEHistory[1] == 0 || KinEHistory[2] == 0 || KinEHistory[3] == 0 || KinEHistory[4] == 0 || KinEHistory[5] == 0)) return true;
	 else return false;
}

Vec3D<> CVX_Sim::GetSumForce(CVX_FRegion* pRegion)
{
	//Vec3D<> ForceSum(0,0,0);

	//Vec3D<> BCpoint;
	//Vec3D<> BCsize = pEnv->pObj->GetLatDimEnv()/2.0;
	//Vec3D<> WSSize = pEnv->pObj->GetWorkSpace();

	//int NumVoxS = Vx.voxelCount();
	//for (int i=0; i<NumVoxS; i++){
	//	BCpoint = pEnv->pObj->GetXYZ(StoXIndexMap[i]);
	//	if (pRegion->GetRegion()->IsTouching(&BCpoint, &BCsize, &WSSize)){
	//		ForceSum += VoxArray[i].GetCurForce(true);
	//		ForceSum -= VoxArray[i].GetExternalForce(); //subract out applied force
	//	}
	//}

	//return ForceSum;

	return Vec3D<>(0,0,0);

	//Vec3D<> ForceSum(0,0,0);
	//Vec3D<> BCpoint;
	//Vec3D<> BCsize = pEnv->pObj->GetLatDimEnv()/2.0;
	//Vec3D<> WSSize = pEnv->pObj->GetWorkSpace();

	//int NumVoxS = NumVox();
	//for (int i=0; i<NumVoxS; i++){
	//	BCpoint = pEnv->pObj->GetXYZ(StoXIndexMap[i]);
	//	if (pRegion->GetRegion()->IsTouching(&BCpoint, &BCsize, &WSSize)){
	//		ForceSum += VoxArray[i].GetCurForce(true);
	//		ForceSum -= VoxArray[i].GetExternalForce(); //subract out applied force
	//	}
	//}

	//return ForceSum;
}

vfloat CVX_Sim::GetSumForceDir(CVX_FRegion* pRegion)
{
	//right now only fixed regions... (forced regions should be zero!)
	//get force only in dircetion of pull!
	Vec3D<> Res = GetSumForce(pRegion);

	Vec3D<> Dir = pRegion->Displace;
	if (Dir.Length2() == 0) return Res.Length(); //return magnitude of no direction...
	else {
		Dir.Normalize();
		return Res.Dot(Dir);
	}

}

Vec3D<> CVX_Sim::GetAvgDisplace(CVX_FRegion* pRegion) //returns the average displacement in x,y,z of a region
{
	//Vec3D<> DispSum(0,0,0);
	//int NumSummed = 0;

	//Vec3D<> BCpoint;
	//Vec3D<> BCsize = pEnv->pObj->GetLatDimEnv()/2.0;
	//Vec3D<> WSSize = pEnv->pObj->GetWorkSpace();

	//int NumVoxS = NumVox();
	//for (int i=0; i<NumVoxS; i++){
	//	BCpoint = pEnv->pObj->GetXYZ(StoXIndexMap[i]);
	//	if (pRegion->GetRegion()->IsTouching(&BCpoint, &BCsize, &WSSize)){
	//		DispSum += VoxArray[i].GetCurPos()-VoxArray[i].GetNominalPosition(); //  GetCurForce(true);
	//		NumSummed++;
	//	}
	//}

	//if (NumSummed == 0) return Vec3D<>(0,0,0);
	//else return DispSum/NumSummed;
	
	return Vec3D<>(0,0,0);
}

void CVX_Sim::SetGravityAccel(float grav)
{
	Vx.setGravity(-grav/9.80665);
}

float CVX_Sim::GetGravityAccel(void)
{
	return -9.80665*Vx.gravity();
}


int CVX_Sim::NumYielded(void)
{
	int NumYieldRet = 0;
	int NumBondNow = Vx.linkCount();
	for (int i=0; i<NumBondNow; i++)
		if (Vx.link(i)->isYielded()) NumYieldRet++;

	return NumYieldRet;
}

int CVX_Sim::NumBroken(void)
{
	int NumBrokenRet = 0;
	int NumBondNow = Vx.linkCount();
	for (int i=0; i<NumBondNow; i++)
		if (Vx.link(i)->isFailed()) NumBrokenRet++;

	return NumBrokenRet;

}



