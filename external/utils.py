"""
Source: https://github.com/AngelRuizMoreno/Jupyter_Dock/blob/main/utilities/utils.py
"""

import py3Dmol
from pymol import cmd
from openbabel import pybel

from rdkit import Chem
from rdkit.Chem import AllChem,rdFMCS, rdMolAlign

from pdbfixer import PDBFixer
from openmm.app import PDBFile

import MDAnalysis as mda
from MDAnalysis.coordinates import PDB

import random, math

import numpy as np

def getbox(selection='sele', extending = 6.0, software='vina'):
    
    ([minX, minY, minZ],[maxX, maxY, maxZ]) = cmd.get_extent(selection)

    minX = minX - float(extending)
    minY = minY - float(extending)
    minZ = minZ - float(extending)
    maxX = maxX + float(extending)
    maxY = maxY + float(extending)
    maxZ = maxZ + float(extending)
    
    SizeX = maxX - minX
    SizeY = maxY - minY
    SizeZ = maxZ - minZ
    CenterX =  (maxX + minX)/2
    CenterY =  (maxY + minY)/2
    CenterZ =  (maxZ + minZ)/2
    
    cmd.delete('all')
    
    if software == 'vina':
        return {'center_x':CenterX,'center_y': CenterY, 'center_z': CenterZ},{'size_x':SizeX,'size_y': SizeY,'size_z': SizeZ}
    elif software == 'ledock':
        return {'minX':minX, 'maxX': maxX},{'minY':minY, 'maxY':maxY}, {'minZ':minZ,'maxZ':maxZ}
    elif software == 'both':
        return ({'center_x':CenterX,'center_y': CenterY, 'center_z': CenterZ},{'size_x':SizeX,'size_y': SizeY,'size_z': SizeZ}),({'minX':minX, 'maxX': maxX},{'minY':minY, 'maxY':maxY}, {'minZ':minZ,'maxZ':maxZ})
    
    else:
        print('software options must be "vina", "ledock" or "both"')


def fix_protein(filename='',addHs_pH=7.4,output='',try_renumberResidues=False):

    fix = PDBFixer(filename=filename)
    fix.findMissingResidues()
    fix.findNonstandardResidues()
    fix.replaceNonstandardResidues()
    fix.removeHeterogens(True)
    fix.findMissingAtoms()
    fix.addMissingAtoms()
    fix.addMissingHydrogens(addHs_pH)
    PDBFile.writeFile(fix.topology, fix.positions, open(output, 'w'))

    if try_renumberResidues == True:
        try:
            original=mda.Universe(filename)
            from_fix=mda.Universe(output)

            resNum=[res.resid for res in original.residues]
            for idx,res in enumerate(from_fix.residues):
                res.resid = resNum[idx]

            save=PDB.PDBWriter(filename=output)
            save.write(from_fix)
            save.close()
        except Exception:
            print('Not possible to renumber residues, check excepton for extra details')
        

def generate_ledock_file(receptor='pro.pdb',rmsd=1.0,x=[0,0],y=[0,0],z=[0,0], n_poses=10, l_list=[],l_list_outfile='',out='dock.in'):
    rmsd=str(rmsd)
    x=[str(x) for x in x]
    y=[str(y) for y in y]
    z=[str(z) for z in z]
    n_poses=str(n_poses)

    with open(l_list_outfile,'w') as l_out:
        for element in l_list:
            l_out.write(element)
    l_out.close()

    file=[
        'Receptor\n',
        receptor + '\n\n',
        'RMSD\n',
        rmsd +'\n\n',
        'Binding pocket\n',
        x[0],' ',x[1],'\n',
        y[0],' ',y[1],'\n',
        z[0],' ',z[1],'\n\n',
        'Number of binding poses\n',
        n_poses + '\n\n',
        'Ligands list\n',
        l_list_outfile + '\n\n',
        'END']
    
    with open(out,'w') as output:
        for line in file:
            output.write(line)
    output.close()



def dok_to_sdf (dok_file=None,output=None):

    """
    dok_to_sdf ( dok_file=None, output=None )

    params:

    dok_file: str or path-like ; dok file from ledock docking

    output: str or path-like ; outfile from ledock docking, extension must be sdf

   """
    out=pybel.Outputfile(filename=output,format='sdf',overwrite=True)

    with open(dok_file, 'r') as f:
        doc=[line for line in f.readlines()]
    
    doc=[line.replace(line.split()[2],line.split()[2].upper()) if 'ATOM' in line else line for line in doc]
    
    start=[index for (index,p) in enumerate(doc) if 'REMARK Cluster' in p]
    finish=[index-1 for (index,p) in enumerate(doc) if 'REMARK Cluster' in p]
    finish.append(len(doc))

    interval=list(zip(start,finish[1:]))
    for num,i in enumerate(interval):
        block = ",".join(doc[i[0]:i[1]]).replace(',','')

        m=pybel.readstring(format='pdb',string=block)
        
        m.data.update({'Pose':m.data['REMARK'].split()[4]})
        m.data.update({'Score':m.data['REMARK'].split()[6]})
        del m.data['REMARK']

        out.write(m)

    out.close()
  

def pdbqt_to_sdf(pdbqt_file=None,output=None):

    results = [m for m in pybel.readfile(filename=pdbqt_file,format='pdbqt')]
    out=pybel.Outputfile(filename=output,format='sdf',overwrite=True)
    for pose in results:

        pose.data.update({'Pose':pose.data['MODEL']})
        pose.data.update({'Score':pose.data['REMARK'].split()[2]})
        del pose.data['MODEL'], pose.data['REMARK'], pose.data['TORSDO']

        out.write(pose)
    out.close()


def get_inplace_rmsd (ref,target):
    
    r=rdFMCS.FindMCS([ref,target])
    
    a=ref.GetSubstructMatch(Chem.MolFromSmarts(r.smartsString))
    b=target.GetSubstructMatch(Chem.MolFromSmarts(r.smartsString))   
    amap=list(zip(a,b))
    
    distances=[]
    for atomA, atomB in amap:
        pos_A=ref.GetConformer().GetAtomPosition (atomA)
        pos_B=target.GetConformer().GetAtomPosition (atomB)
        coord_A=np.array((pos_A.x,pos_A.y,pos_A.z))
        coord_B=np.array ((pos_B.x,pos_B.y,pos_B.z))
        dist_numpy = np.linalg.norm(coord_A-coord_B)        
        distances.append(dist_numpy)
        
    rmsd=math.sqrt(1/len(distances)*sum([i*i for i in distances]))
    
    return rmsd

def get_scaffold_based_conformers(smiles=None, anchor=None, num_confs=None, output=None, rmsd_threshold=0.75):
    mol = Chem.MolFromSmiles(smiles,sanitize=True)
    mol=Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    constrain = Chem.SDMolSupplier(anchor,sanitize=True)[0]

    r = rdFMCS.FindMCS([mol, constrain])
    a = mol.GetSubstructMatch(Chem.MolFromSmarts(r.smartsString))
    b = constrain.GetSubstructMatch(Chem.MolFromSmarts(r.smartsString))
    amap = list(zip(a, b))
    coors = dict()

    for a,b in amap:
        coors[a] = constrain.GetConformer().GetAtomPosition(b)

    w = Chem.SDWriter(output)

    mol.UpdatePropertyCache()
    constrain.UpdatePropertyCache()

    confs = AllChem.EmbedMultipleConfs(mol,
        numConfs=int(num_confs),
        coordMap=coors,
        pruneRmsThresh=0.75,
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True)

    for element in confs:
        Chem.SanitizeMol(mol)
        rmsd = AllChem.GetBestRMS(mol,constrain,element,0,map=[list(amap)])
        if rmsd<=float(rmsd_threshold):
            w.write(mol, confId=element)
    w.close()
