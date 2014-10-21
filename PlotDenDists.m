n = length(denDists(:,1));
% Get denDists from python newVectors.getDenDists(cons)
DoriDspdDendDif = [] % <Dori><Dspeed><DendDist><DiffDendrite(color)>
for i = 1:n
    sk1= denDists(i,2);
    sk2 = denDists(i,3);
    if sk1 ~= sk2
        ori1 = EMidOriRGBneuronIDSFTFspeed(EMidOriRGBneuronIDSFTFspeed(:,1) == sk1,2);
        ori2 = EMidOriRGBneuronIDSFTFspeed(EMidOriRGBneuronIDSFTFspeed(:,1) == sk2,2);
        dori = abs(ori1-ori2);
%         if dori >90 
%             dori = 180 - dori;
%         end
        
        spd1 = EMidOriRGBneuronIDSFTFspeed(EMidOriRGBneuronIDSFTFspeed(:,1)== sk1, 9);
        spd2 = EMidOriRGBneuronIDSFTFspeed(EMidOriRGBneuronIDSFTFspeed(:,1)== sk2, 9);
        dspd = abs(spd1-spd2)
        DoriDspdDendDif = [DoriDspdDendDif; [dori, dspd, denDists(i,4), denDists(i,5)]];
    end
end
figure
subplot(2,1,1)
hold on

scatter(DoriDspdDendDif(:,1),DoriDspdDendDif(:,3),20,DoriDspdDendDif(:,4))
title('dori vs dist between connectors')
b=subplot(2,1,2)
hold on
scatter(DoriDspdDendDif(:,2),DoriDspdDendDif(:,3),20,DoriDspdDendDif(:,4))
title('dspd vs dist between connectors')
