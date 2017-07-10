
------------------------------
-- library
------------------------------

require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'

------------------------------
-- function
------------------------------

function newmodel()

	-- 5-class problem
	local noutputs = 8

	-- load pretrain model
	local model_dir = 'pretrain'
	local model = torch.load(paths.concat(model_dir, 'resnet-34.t7'))

	-- remove last full connect
	model:remove(model:size())

	-- remove view
	model:remove(model:size())

	-- remove average pooling
	model:remove(model:size())

	-- add output module
	model:add(cudnn.SpatialConvolution(512, noutputs, 1, 1))
	model:add(nn.LogSoftMax())

	return model
end

------------------------------
-- main
------------------------------

model = newmodel()
-- print(model:size())

-- loss function
criterion = nn.SpatialClassNLLCriterion()
print '==> here is the loss function:'
print(criterion)

model:cuda()
criterion:cuda()

-- samp = torch.Tensor(4, 3, 224, 224):uniform()
-- samp = samp:cuda()

-- output = model:forward(samp)
-- print(output:size())
