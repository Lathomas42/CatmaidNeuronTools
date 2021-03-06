load('COMList.mat')
load('adjs.mat')
load('skels.mat')

n = length(COMList(:,1))
ConCOMDists = [];
UnConCOMDists = [];
for i = 1:(n-1)
    i_x = COMList(i,2);
    i_y = COMList(i,3);
    i_z = COMList(i,4);
    i_ap = COMList(i,5);
    i_ori = COMList(i,7);
    i_spd = COMList(i,8);
    for j = (i+1):n
        j_x = COMList(j,2);
        j_y = COMList(j,3);
        j_z = COMList(j,4);
        j_ap = COMList(j,5);
        j_ori = COMList(j,7);
        j_spd = COMList(j,8);
        
        d_x = i_x - j_x;
        d_y = i_y - j_y;
        d_z = i_z - j_z;
        d_ori = abs(i_ori - j_ori);
        if d_ori > 180
            d_ori
        end
        if(d_ori > 90)
            d_ori = 180-d_ori;
        end
        d_spd = abs(i_spd - j_spd);
        
        dist = (d_x^2 + d_y^2 + d_z^2)^.5;
        
        if (AdjNoInhNoAxNew(i,j)+AdjNoInhNoAxNew(j,i) ~= 0)
            ConCOMDists = [ConCOMDists;[dist, d_ori, d_spd, i_ap+j_ap]] ;
        else
            UnConCOMDists = [UnConCOMDists;[dist, d_ori, d_spd, i_ap+j_ap]];
        end
    end
end
figure 
subplot(2,1,1)
hold on
title('Con DORI')
scatter(ConCOMDists(:,2),ConCOMDists(:,1),20,ConCOMDists(:,4))
subplot(2,1,2)
hold on
title('Con DSPD')
scatter(ConCOMDists(:,3),ConCOMDists(:,1),20,ConCOMDists(:,4))

figure
subplot(2,1,1)
hold on
title('UnCon DORI')
scatter(UnConCOMDists(:,2),UnConCOMDists(:,1),20,UnConCOMDists(:,4))
subplot(2,1,2)
hold on
title('UnCon DSPD')
scatter(UnConCOMDists(:,3),UnConCOMDists(:,1),20,UnConCOMDists(:,4))