clear 
clc

%%
rad_Zed = [-0.15 -0.92 -0.34]; % ZYX格式，使用ZedDemo读出
RZed = eul2rotm(rad_Zed, 'ZYX');
R_x180 = [1,0,0;
            0,-1,0;
            0,0,-1];
R = R_x180 * RZed * R_x180;
T = [1, 0,0, 0; 
     0, 0,1, 0;
     0,-1,0, 2;
     0,0,0,  1] * [R, zeros(3,1);zeros(1,3),1];