require 'loadcaffe'
require 'cudnn'
require 'nn'
require 'cutorch'
require 'paths'
require 'image'
require 'mattorch'
require 'cunn'


-- Import trained model 
model=torch.load('./trainedModels/resnet-152.t7')


--Remove last layers
model:remove(11)
model:remove(10)

--Print model
--print(model)


--Data directories. 
data_path = "./MPII/jpgAllFrames/"
folder_names={}


function preprocessImage(img) 

    -- Extrqact the center crop 
    height = img:size(2)
    width = img:size(3)
    img = image.scale(img,256,256)
    img = image.crop(img,16,16,240,240) --will return 3,224,224 image. 

    mean = { 0.485, 0.456, 0.406 }
    std = { 0.229, 0.224, 0.225 }
 
    
    for i=1,3 do 
        img[{{i},{},{}}] = img[{{i},{},{}}] - mean[i]
        img[{{i},{},{}}] = img[{{i},{},{}}]/std[i]
    end
    
    return img
end

g=1

dirNames = torch.load('dirNames.t7')

for d1=84,85 do
    print("Processing Movie : " .. d1 .. "   " , dirNames[d1])
    os.execute('mkdir ./features/MPII/resNet/' .. dirNames[d1])
    t1 = os.time()
    for subdir in paths.iterdirs(data_path .. dirNames[d1]) do
        file_names = {}
        
        for file in paths.iterfiles(data_path .. dirNames[d1] .. '/' ..  subdir .. '/.') do
            table.insert(file_names,file)
        end
        
        table.sort(file_names)
        num_files = #file_names
        
        --if num_files>500 then
        --    print("Problem in subdirectory  ", subdir) 
        --    imgData = torch.rand(#file_names/2,3,224,224) 
        --    num_files = num_files/2
        --else 
        --    imgData = torch.rand(#file_names,3,224,224) 
        ---end
                  
        --for j=1,num_files do   
        --     temp_data = image.load(data_path .. dirNames[d1] .. '/' .. subdir .. '/' ..  file_names[j],3,'float')
        --     temp_data = preprocessImage(temp_data)
        --     imgData[j] = temp_data
        --end
        --print("Size of image data", #imgData)
             
        --print(file_names)
        
        
        videoData = torch.rand(num_files,2048)
        batch = 20
        num_batch = math.floor(num_files/batch)
        if num_files%batch~=0 then 
            num_batch = num_batch +1
        end 
        for k=1,num_batch do
            startb=(k-1)*batch+1
            endb = startb+batch
            if endb > num_files then endb = num_files end
            imgData = torch.rand(endb-startb+1,3,224,224)
            for b1=startb,endb do
                temp_data = image.load(data_path .. dirNames[d1] .. '/' .. subdir .. '/' ..  file_names[b1],3,'float')
                temp_data = preprocessImage(temp_data)
                imgData[b1-startb+1] = temp_data
            end
            --batchdata = imgData[{{startb,endb},{},{},{}}]  
            
            op = model:forward(imgData:cuda())
            op = torch.reshape(op,imgData:size(1),2048)
  
            videoData[{{startb,endb},{}}] = op:float()      
        


        end 
        print("Movie: " .. d1  .. "  " ..  dirNames[d1] .. "  's clip" .. subdir .. " has " .. num_files .. "  number of frames")
   

        
  
        --print('./features/MPII/resNet/'.. dir .. '/' .. subdir .. ".mat")
       
        mattorch.save('./features/MPII/resNet/'.. dirNames[d1] .. '/' .. subdir .. ".mat",videoData)

    end
    print ("Execution Time : ", os.difftime(os.time(),t1))
    g=g+1 
        
 
end


