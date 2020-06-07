p = 1;

for imageN = 1:482
    name = ['Image',num2str(imageN),'.jpg'];
    %normImage = im2double(name);
    I = imread(name);%I = imread('~/Desktop/StormsOf/Jupiter/Image%.jpg');
    %I =
    %Im = imadjust(I);
    data(p,:) = reshape(I,1,28*28);
    %data(p,:) = I;
    p = p + 1;
    
end
csvwrite('images.csv',data)