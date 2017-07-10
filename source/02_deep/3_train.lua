
------------------------------
-- library
------------------------------

require 'torch'
require 'xlua'
require 'optim'
require 'cunn'

require 'lib/get_random_sample'
require 'lib/get_positive_sample'

------------------------------
-- main
------------------------------

-- Retrieve parameters and gradients:
if model then
	parameters, gradParameters = model:getParameters()
end

-- optimizer
print '==> configuring optimizer'
optimState = {
	learningRate = opt.learningRate,
	weightDecay = opt.weightDecay,
	momentum = opt.momentum,
	learningRateDecay = 1e-7
}
optimMethod = optim.adam

------------------------------
-- function
------------------------------

function byte_and(tensor_a, tensor_b)
	local function f_and(xx, yy) return xx * yy end
	local ret = tensor_a:clone()
	ret:map(tensor_b, f_and)
	return ret
end

function generate_target_label(output)

	local target = torch.ByteTensor(1,7,7):fill(1)

	-- 1: nothing
	-- 2: pups
	-- 3: juveniles
	-- 4: adult_females
	-- 5: adult_females & pups
	-- 6: adult_females & juveniles
	-- 7: subadult_males
	-- 8: adult_males

	target:cmax(output[5] * 2)
	target:cmax(output[4] * 3)
	target:cmax(output[3] * 4)
	target:cmax(byte_and(output[3], output[5]) * 5)
	target:cmax(byte_and(output[3], output[4]) * 6)
	target:cmax(output[2] * 7)
	target:cmax(output[1] * 8)

	return target
end

function train()

	-- number of data
	local train_size = 5000

	-- epoch tracker
	epoch = epoch or 1

	-- set model to training mode
	model:training()

	-- loss variable
	local log_loss = 0

	-- do one epoch
	print(sys.COLORS.cyan .. '==> training on train set: # ' .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	for t = 1,train_size,opt.batchSize do

		-- disp progress
		xlua.progress(t, train_size)

		local local_batchSize = math.min(opt.batchSize, train_size - t + 1)

		-- create mini batch
		local inputs = torch.Tensor(local_batchSize, 3, 224, 224)
		local targets = torch.Tensor(local_batchSize, 7, 7)

		for local_count, i in ipairs( tablex.range(t, t+local_batchSize-1) ) do
			-- load new sample
			local input, output
			input, output, _ = get_random_sample(train_list, patch_size)
			-- input, output, _ = get_positive_sample(patch_size)

			local target = generate_target_label(output)

			-- paths.mkdir('_input')
			-- image.save('_input/' .. i .. 'input.png', input)
			-- for c = 1,5 do
			-- 	image.save('_input/' .. i .. 'output_' .. labels[c] .. '.png', output[c]:float())
			-- end

			-- [1] flip
			if math.random() > 0.5 then
				input = image.hflip(input)
				target = image.hflip(target)
			end

			-- [2] rotate
			for r = 1,math.random(4) do
				input = image.hflip(input):transpose(2, 3)
				target = image.hflip(target):transpose(2, 3)
			end

			inputs[{{local_count}}] = input
			targets[{{local_count}}] = target

			-- print(generate_target_label(output))
		end

		inputs = inputs:cuda()
		targets = targets:cuda()

		-- create closure to evaluate f(X) and df/dX
		local feval = function(x)
			-- get new parameters
			if x ~= parameters then
				parameters:copy(x)
			end

			-- reset gradients
			gradParameters:zero()

			-- estimate output
			local output = model:forward(inputs)

			-- f is the average of all criterions
			local f = criterion:forward(output, targets)
			log_loss = log_loss + f * local_batchSize

			-- estimate df/dW
			local df_do = criterion:backward(output, targets)
			model:backward(inputs, df_do)

			-- normalize gradients
			gradParameters:div(local_batchSize)

			-- return f and df/dX
			return f,gradParameters
		end

		-- optimize on current mini-batch
		optimMethod(feval, parameters, optimState)
	end
	xlua.progress(train_size, train_size)

	-- calc train score
	train_score = log_loss / train_size
	print('\ttrain_score: ' .. string.format('%.4f', train_score))

	-- save/log current net
	local file_name = 'model_epoch' .. epoch .. '.net'
	local file_path = paths.concat(opt.path_models, file_name)
	paths.mkdir(sys.dirname(file_path))
	-- print('\tsaving model to '..file_path)
	torch.save(file_path, model:clearState())

	-- next epoch
	epoch = epoch + 1
end
