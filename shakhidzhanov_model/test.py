t = linspace(0, 3000, 1e6)
u0 = [Ca_cyt_0, Ca_dts_0, Ca_mit_0, Serca2bE1_0, Serca2bE1Ca_0, Serca2bE1CaCa_0, Serca2bE1CaCaP_0, Serca2bE2CaCaP_0,
      Serca2bE2P_0, Serca2bE2_0, IP3_0, IP3r_0, IP3Ro_0, IP3Ra_0, IP3Ri1_0, IP3Ri2_0, IP3Rs_0, Top_0, Del_psi_0,
      Receptors, PAR1_Tr_0, PAR1_Tr_G_0, PAR1a_Tr_0, PAR1a_Tr_G_0, GqGTP_0, GqGDP_0, Gqbg_0, G_0, P2Y12_0,
      P2Y12_ADP_0, P2Y12_ADP_G_0, P2Y12a_ADP_0,  P2Y12a_ADP_G_0, GiaGTP_0, GiaGDP_0, Gbg_0, G1_0, cAMP_0, AMP_0,
      PDE2_0, PDE3_0, PDE2a_0, PDE3a_0, C_0, R_0, R_C_0, R2_cAMP_0, R2_cAMP_C_0, RA_cAMP_0, RB_cAMP_0, RB_cAMP_C_0,
      Thrombin, ADP]
u = odeint(dotu, u0, t)
