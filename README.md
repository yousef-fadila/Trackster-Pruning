# Trackster-Pruning
Trackster Pruning at the CMS High-Granularity Calorimeter

The High Luminosity LHC (HL-LHC) will collect 10 times more integrated
luminosity than the LHC, posing significant challenges for radiation
tolerance and event pileup on detectors, especially for forward
calorimetry. As part of its HL-LHC upgrade program, the CMS collaboration
is designing a High Granularity Calorimeter to replace the existing
endcap calorimeters. It features unprecedented transverse and
longitudinal segmentation for both electromagnetic (ECAL) and hadronic
(HCAL) compartments. This will facilitate particle-flow calorimetry,
where the fine structure of showers can be measured and used to enhance
pileup rejection and particle identification, while still achieving
excellent energy resolution. The intrinsic high-precision timing
capabilities of the silicon sensors will add an extra dimension to event
reconstruction, especially in terms of pileup rejection. The current
HGCAL reconstruction produces hundreds of tracksters per simulated proton-proton collision. 

Tracksters are Direct Acyclic Graphs representing topologically connected hits belonging to a single physics objects.
However, in the presence of high pile-up collisions, the tracksters could be contaminated by unrelated hits. 
<b>The objective of the project is to design and deploy a Neural Networks to prune the tracksters from pile-up contamination within milliseconds.</b>
