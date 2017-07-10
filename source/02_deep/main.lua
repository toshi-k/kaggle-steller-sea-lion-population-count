
------------------------------
-- library
------------------------------

require 'nn'
require 'cunn'

------------------------------
-- setting
------------------------------

cmd = torch.CmdLine()
cmd:text('Options:')
-- global:
cmd:option('-seed', 0, 'fixed input seed for repeatable experiments')
-- path:
cmd:option('-path_models', '_models', 'subdirectory to save models')
cmd:option('-path_result', 'submission_pre', 'subdirectory to result file')
-- training:
cmd:option('-learningRate', 1e-4, 'learning rate at t=0')
cmd:option('-weightDecay', 0, 'weight decay')
cmd:option('-momentum', 0, 'momentum')
cmd:option('-batchSize', 24, 'mini-batch size')
cmd:text()
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')

------------------------------
-- main
------------------------------

-- set seed ------------------

seed = opt.seed
print(string.format('==> set seed: %d', seed))
math.randomseed(seed)
torch.manualSeed(seed)
cutorch.manualSeed(seed)

-- start timer ---------------

timer = torch.Timer()

-- load files ----------------

dofile "1_data.lua"
dofile "2_model.lua"
dofile "3_train.lua"
dofile "4_test.lua"
dofile "5_valid.lua"

-- set patch parameters ------

patch_param_input = {}
patch_param_output = {7, 1, math.random(7)-1, math.random(7)-1}
for _, v in ipairs(patch_param_output) do
	table.insert(patch_param_input, v * 32)
end
print('==> Set patch parameters')
print(patch_param_output)

-- train ---------------------

max_itr = 15

for Itr = 1,max_itr do
	train()

	if Itr % 5 == 0 then
		valid(train_list, true, true)
	end

	valid(valid_list, Itr < 5, false)
end

-- test ----------------------

test()

-- display elapsed time ------

elapsed = timer:time().real
print('Elapsed Time: ' .. xlua.formatTime(elapsed))
