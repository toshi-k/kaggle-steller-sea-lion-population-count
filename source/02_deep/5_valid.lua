
------------------------------
-- library
------------------------------

require 'torch'
require 'xlua'
require 'gnuplot'

require 'lib/patch'
require 'lib/window'
require 'lib/extract'

------------------------------
-- function
------------------------------

-- Moore–Penrose pseudo-inverse
-- https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse
function inverse(mat)
	return torch.mm(torch.inverse(torch.mm(mat:t(), mat)), mat:t())
end

function valid(local_valid_list, is_update_coef, is_train)

	-- set model to evaluate mode
	model:evaluate()

	local image_dir = '../../dataset/Train'
	local dot_dir = '../../dataset/TrainDotted'

	local num_valid = #local_valid_list
	local features = torch.Tensor(5, num_valid, 3)
	local truth = torch.Tensor(5, num_valid)

	if is_train then
		print(sys.COLORS.yellow .. '==> validating on train set:')
	else
		print(sys.COLORS.green .. '==> validating on valid set:')
	end

	-- validate over valid data
	for t, valid_num in ipairs(local_valid_list) do

		-- disp progress
		xlua.progress(t, #local_valid_list)

		-- file path for valid image
		local valid_img_path = paths.concat(image_dir, valid_num .. '.jpg')
		local dotted_img_path = paths.concat(dot_dir, valid_num .. '.jpg')

		-- load valid image
		local valid_img = image.load(valid_img_path)
		local dotted_img = image.load(dotted_img_path)

		local mask = dotted_img:sum(1)
		mask = mask:ge(0.01):float()

		-- masking
		for c = 1,valid_img:size(1) do
			valid_img[{{c}}]:cmul(mask)
		end

		-- predict function is defined in 4_test.lua
		local valid_img_pred = predict(valid_img)

		paths.mkdir(paths.concat('_save', 'valid'))
		if t < 20 then

			image.save(paths.concat('_save', 'valid', string.format('valid_%s_zmask.jpg', valid_num)), mask)

			for c = 1,5 do
				local img_save = valid_img_pred[c]
				local file_name = string.format('valid_%s_%s.jpg', valid_num, labels[c])
				image.save(paths.concat('_save', 'valid', file_name), img_save)
			end
		end

		for c = 1,5 do
			features[{c,t,{}}] = extract(valid_img_pred[c])
		end
		truth[{{},t}] = train_csv[valid_num]
	end
	xlua.progress(#local_valid_list, #local_valid_list)

	-- coefs must be global variable to use in test function
	coefs = coefs or torch.zeros(5, 3)
	local rmses = torch.Tensor(5)

	paths.mkdir('_figure')
	for c = 1,5 do

		local F = features[c]
		local y = truth[c]

		if is_update_coef then
			local invF = inverse(F)
			local coef = torch.mv(invF, y)
			coefs[c] = coef
		end

		local predict = torch.mv(F, coefs[c])
		predict[predict:le(0)] = 0

		local diff = predict - y

		gnuplot.epsfigure(paths.concat('_figure', string.format('epoch_%d_%s.eps', epoch-1, labels[c])))
		gnuplot.plot({'Predict', predict, y, '+'})
		gnuplot.xlabel('Predict')
		gnuplot.ylabel('Truth')
		gnuplot.plotflush()

		local rmse = math.sqrt(torch.mean(torch.pow(diff, 2)))
		rmses[c] = rmse
	end

	if is_train then
		train_score = torch.mean(rmses)
		print('\ttrain_score: ' .. string.format('%.4f', train_score))
	else
		valid_score = torch.mean(rmses)
		print('\tvalid_score: ' .. string.format('%.4f', valid_score))
	end
end
