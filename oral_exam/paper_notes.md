% This is a rough outline of the oral write-up

# Rolling Project
## Introduction

1. Description of the priming project
	- Why is hemocompatibility important?
		Cardiovascular disease is one of the leading causes of death in the U.S, and a common treatment for these diseases is to implant medical devices into the blood stream. For example, stents (an expandable solid mesh) are often used to treat stenotic arteries. However, the introduction of a foreign material into the blood stream will cause thrombosis unless the material is treated somehow, and while these devices have been effective in saving lives, they are still far from perfect. Patients with these implanted devices still must be placed on anticoagulants, as they have a higher risk of a thrombotic event even when the device is functional and surgery is successful (cit: ref 9 in Colin's thesis--Cannegieter et. al., 1994). 
		
	- What has been done so far in hemocompatibility research, and what is Vlado's group trying to add? (local to nonlocal)
		Most hemocompatibility studies focus on local interactions and effects of the material on platelets. However recent work (cit: Dr. Hlady's work) has shown that nonlocal effects are also important in understanding platelet interactions with implanted biomaterials. In particular they have shown that platelet interactions with immobilized agonists may not cause the platelets to adhere at the site of interaction, but can nonetheless prime them for downstream adhesion and full activation. One possible example of this in a vascular graft is shown in Figure #. At either end of the implanted device, native tissue must be joined with the artificial device. At these points along the vessel wall the tissue is inflamed and anastomotic (source?), which could expose platelet agonists on the surface of the wall. Additionally the increased shear rate in the stenotic regions could act as a platelet agonist (source?). Therefore while platelets may not bind to a single inflamed region of the vessel, the upstream region may prime platelets for adhesion, and then more readily bind to the downstream inflamed region. It is also possible that the inflamed tissue or the biomaterial may not cause platelets to bind locally, but nonetheless partially activate platelets and cause them to adhere downstream somewhere in the circulation. 
		
	- Describe experimental setup, ongoing experiments, ...
		In order to investigate nonlocal effects of priming, Dr. Hlady's group has designed a microfluidic assay with two regions printed  with platelet agonists (see Figure whatever). The upstream agonist region will be called the priming region, and the downstream region with agonist will be called the capture region. In previous research, Dr. Hlady's group has found that all 3 immobilized platelet agonists tested (vWF, collagen, and fibrinogen) in the upstream region increased adhesion of platelets in the capture region relative to a nonreactive control (Colin's thesis, Corum paper). Also, recent unpublished work has shown similar results when the upstream agonist printing is replaced by a stenotic region with higher wall shear rates (reference Shek's work).
	
2. Contextualize rolling within blood clotting
	- Process of clotting begins when agonists (vWF and collagen) are exposed on the ECs
		In order for a platelet aggregate to form, two processes (wording?) need to occur: platelet adhesion to the vessel wall, and cohesion among platelets. In normal hemostasis, adhesion is facilitated by two surface-immobilized proteins: collagen embedded in the sub-endothelial matrix, and von Willebrand factor (vWF) which can either be bound to collagen in the SE matrix or expressed by endothelial cells in pathological circumstances (see refs in Fogelson and Neeves, 2015, pp. 384 & 390). 
		
	- Platelets bind to these agonists with fast receptors first (GP1b and GPVI), and then later with slow receptors (\alpha{IIb} \beta{3} and \alpha{2} \beta{1}).
		Platelets constitutively express GP1b and GPVI receptors for vWF and collagen respectively (see refs in Fogelson and Neeves 2015, p. 384). These receptors have fast binding and unbinding kinetics, and are therefore able to form bonds with their surface-immobilized ligands even when the platelets are in flows with high wall shear rates. When these bonds form and are stretched beyond their rest length, they exert a force in opposition to the flow and slow the platelet down. Therefore the motion of a platelet along a wall with immobilized agonists is a function of the biochemical interactions of receptors and ligands as well as the fluid forces exerted on the platelet. 
		
	- The fast receptors trigger intracellular signaling cascades that result in platelet activation and activation of the slow receptors.
		These receptors are more than simple physical links between the platelet and surface. When they are bound with a ligand, they initiate intracellular signaling cascades that are responsible for triggering release of intracellular calcium and activating phosphatidylinositide-3-kinase (PI3K) which are two crucial events in platelet activation (Bye paper; look at Fogelson and Neeves; Du paper?). These activation pathways ultimately terminate in a suite of responses that are collectively called "platelet activation," including granule secretion, TxA2 synthesis, cytoskeletal rearrangements, and activation of integrins \alpha{IIb}\beta{3} and \alpha{2}\beta{1}. These integrins are receptors for fibrinogen/vWF and collagen, respectively. They are constitutively expressed on the surface of the platelet (check this!), but on unactivated platelets they exist mostly in their low-affinity conformation (sources?).
		
		In general, integrins form a large group of transmembrane receptors that are primarily involved in the adhesion of cells to extracellular matrix (ECM) and cell locomotion. They have a large extracellular domain called the ectodomain with a hinge. In the low affinity conformation, the ectodomain is bent at the hinge and the ligand binding domain is at least partially blocked. When switching from low-affinity to high-affinity conformations the integrin ectodomain extends at the hinge to reveal the ligand binding site (sources). Note I've avoided using the words "inactive" and "active." This is intentional: there is evidence that integrins retain some ability to form bonds with their ligands when in the low-affinity conformation (sources), and experiments showing that resting platelets can bind to fibrinogen (sources) also suggest this is possible.
		
		Once integrins are bound and in their high-affinity conformation, they can activate signal transduction pathways to initiate the formation of clusters of integrins that mediate firm adhesion of a cell to ECM (sources), along with other responses. 
	
	- Once platelets are activated, they can bind firmly to the wall, release soluble platelet agonists which causes a platelet aggregate to form, and release thrombin which results in the formation of a fibrin gel which ultimately stabilizes a clot. 

3. Description of cell rolling, a single cell interacting with the surface
	- Should mention margination in here somewhere.
	- Cell rolling is the important first part of the process of platelet activation.
	- There are three main insoluble platelet agonists: vWF, collagen, and fibrinogen.
	- Two important constitutively active platelet receptors are GP1b (binds to vWF) and GPVI (binds to collagen). These receptors have fast association and dissociation and mediate platelet rolling along a vessel wall.
	- There are also constitutively expressed integrins \alpha{IIb} \beta{3} and \alpha{2} \beta{1}, however they are in their low-affinity conformation.
	

## Mathematical Rolling Model
### Problem Description

1. Geometry and Physics

Assume we have a circular rigid platelet (of radius $R$) rolling and translating in shear flow with shear rate $\gamma$ adjacent to a wall. The platelet translates parallel to the wall at speed $V$, and rolls at angular velocity $\Omega$. Because the circle is always translating parallel to the wall, there is a fixed vertical distance $d$ between the wall and the closest point on the circle. Define the height $h$ of the platelet to be the distance from the wall to the center of the platelet ($h = d + R$). The fluid forces exerted on the platelet are a function of $h$, $R$, $\gamma$, and $V$, as well as the fluid viscosity ($\mu$) which is constant across all experiments. We assume that the platelet is moving in a Stokes flow, meaning that the inertial terms are negligible and force balance must be satisfied on the platelet at all times.

Due to the linearity of Stokes equations, there is a linear relationship between platelet velocity and fluid force. In a general 3D Stokes flow, this relationship is given by a 6x6 matrix equation $equation$ where $U$ is a 6x1 vector containing the 3 translational velocities and 3 rotational velocities, and $F$ is a 6x1 vector containing the 3 forces and 3 torques. However in the current model we've made a number of simplifying assumptions:
	1. The flow is a 2D Stokes flow, eliminating all forces and translational velocities along the dimension going into and out of the page. This also eliminates rotations and torques about the two other dimensions.
	2. The platelet remains at a constant separation distance from the wall, eliminating all vertical motion (and consequently we ignore all vertical forces imposed by bonds between the platelet and wall).
	3. The translational velocity is decoupled from torque, and vice versa. (Is this equivalent to assuming the platelet is sitting in a constant flow?)
Therefore we end up with two decoupled linear equations relating the horizontal force ($fh$) to the translational velocity ($V$) and relating the torque ($\tau$) to the rotational velocity ($\Omega$):
Equations. ***Note: I will need to expand on this more when I add equations to the paper. I also forgot to discuss the applied fluid velocities.***

We identify bonds by their two attachment points on the wall and on the platelet surface, so each bond has two coordinates associated with it for each of its endpoints. Points on the wall are given by the coordinate $x$, defined to be the horizontal distance from the center of the circle, and points on the platelet surface are given by the coordinate $\theta$, defined to be the angle formed by the receptor, the center of the circle, and the closest point on the circle to the wall (Figure something). With this definition of $x$ and $\theta$, the distance from a ligand at $x$ on wall and a receptor $\theta$ on the platelet surface is given by the equation
Equation. 

2. Biology

The platelet surface is covered with receptors at an angular density of $N_T$ receptors/radian, and bonds can form between points on the surface of the circle and points on the wall. The number of ligand binding sites on the substrate is assumed to be in excess. We assume that bonds act like Hookean springs with a rest length of 0, so the force exerted by a bond is proportional to its length and the proportionality constant is the stiffness of the bond. Formation and dissociation rates are distance and force dependent, respectively. We use the Bell model (source) to express the bond  dissociation rate as a function of force, and the (something) model to express the bond formation rate as a function of the distance between a receptor and ligand. This gives us the following equations for formation rate and dissociation rate:
Equations.

In order to calculate $fh$ and $\tau$ for an existing set of bonds between the platelet and wall, we simply have to add up the individual contributions to the force and torque from each bond. The force and torque generated by an individual bond can be derived from the geometry of the model along with the assumption that the force vector $F$ points in the same direction as the bond with magnitude proportional to the bond length. I'll leave out the derivation. The horizontal force generated by a single bond ($fhs) is given simply by
Equation,
while the torque generated by a single bond is given by the slightly more complicated
Equation.

### Deterministic PDE model

If we assume that there is a continuous distribution of bonds between the platelet and the wall, we can define the function $n(x, \theta, t)$ which gives the density of bonds at time $t$ between points $x$ and $\theta$ on the wall and platelet, respectively. Bonds advect in $x$ with velocity $V$ and they advect in $\theta$ with velocity $\Omega$. Bonds form at the rate given in equation (something) and saturate in $\theta$ with a maximum density $Nt$. Finally, bonds break at the force-dependent rate given in equation (something). Putting all of this together gives us the following PDE definition of $n$:
Equation.

This equation can't yet be solved for $n$, because there are still 2 unknowns in it: $V$ and $\Omega$. Recall from above that $V$  and $\Omega$ are found by balancing the fluid and bond forces acting on the platelet (equations (something)). First, we have to calculate the total force $fh$ and torque $tau$ generated by the distribution of bonds $n$:
Equations.

Once we have $fh$ and $\tau$, the platelet velocities $V$ and $\Omega$ are given by the equations: Equations.

Now we have 3 equations in 3 unknowns and this is a closed system that can (almost) be solved simultaneously for $n$, $V$, and $\Omega$. We still need to define the domains of all three variables, and give boundary conditions for $n$ before we can solve this system. Assume that bonds can only attach to the lower half-circle of the platelet surface, while bonds can form at any point along the wall. Thus we are solving $n$ in the domain $something$. We need to enforce boundary conditions at the upstream (with respect to variables $x$ and $\theta$) ends of the $x$ and $\theta$ domains. Because we assume there are no bonds attached to the upper half of the platelet, and because the equilibrium concentration of bonds $something rightarrow 0$ as $L \rightarrow \infty$, we must set homogeneous Dirichlet boundary conditions on both $x$ and $\theta$. Finally we just set the initial condition to be $something$. 

In order to eliminate some parameters, we nondimensionalize this model by introducing nondimensional coordinates $x = Rz$ and $s = koff t$ (note that $\theta$ is already dimensionless) and nondimensional functions $R n = Nt m$, $V = R koff v$, and $\Omega = koff \omega$. The details of the nondimensionalization are included in Appendix (something), but the final nondimensional set of equations is 
Equations.

The nondimensional bond length $\ell$ is given by $something$, and the nondimensional force $fhprime$ and torque $\tauprime$ are given by $something$. A complete list of the nondimensional coordinates, functions, and parameters with their definitions are given in Table (something).

### Stochastic models
does it make more sense to change the order of these?

## Results
what results do I have?

## Future directions

## Appendix 1: Nondimensionalization

## Appendix 2: Numerical Schemes

## Appendix 3: Parameter Estimates
