
------------------------------
-- library
------------------------------

require 'torch'
require 'xlua'

require 'lib/patch'
require 'lib/window'
require 'lib/extract'

------------------------------
-- function
------------------------------

function predict(img)

	local height = math.floor(img:size(2)/32) * 32
	local width = math.floor(img:size(3)/32) * 32
	img = img[{{}, {1,height}, {1,width}}]

	local num_channel = 8

	local result_size = img:size()
	local result_dummy = torch.Tensor(num_channel, img:size(2)/32, img:size(3)/32)

	local data = img2patch({img}, patch_param_input)
	local data_pred = torch.Tensor(data:size(1), num_channel, 7, 7)

	for i = 1,data:size(1) do
		local input = data[{{i}}]:cuda()
		local output = model:forward(input):float()
		data_pred[{{i}}] = output
	end
	data_pred:exp()

	local img_pred = patch2img(data_pred, {result_dummy}, patch_param_output)[1]
	-- 1: nothing
	-- 2: pups
	-- 3: juveniles
	-- 4: adult_females
	-- 5: adult_females & pups
	-- 6: adult_females & juveniles
	-- 7: subadult_males
	-- 8: adult_males

	local output = torch.Tensor(5, img_pred:size(2), img_pred:size(3))

	output[{{1}}] = img_pred[8]
	output[{{2}}] = img_pred[7]
	output[{{3}}] = img_pred[4] + img_pred[5] + img_pred[6]
	output[{{4}}] = img_pred[3] + img_pred[6]
	output[{{5}}] = img_pred[2] + img_pred[5]

	return output
end

function test()

	-- set model to evaluate mode
	model:evaluate()

	local image_dir = '../../dataset/Test'

	paths.mkdir('../../submission/' .. opt.path_result)

	local file_name = string.format('submission_s%d_train%.2f_valid%.2f.csv', opt.seed, train_score, valid_score)
	local path = paths.concat('../../submission', opt.path_result, file_name)
	local fp = io.open(path, "w")

	local headers = {'test_id', 'adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups'}
	local headwrite = table.concat(headers, ',')
	fp:write(headwrite .. '\n')

	-- test over test data
	print(sys.COLORS.red .. '==> testing on test set:')
	for t, test_num in ipairs(test_list) do

		-- disp progress
		xlua.progress(t, #test_list)

		-- file path for test image
		local test_img_path = paths.concat(image_dir, test_num .. '.jpg')

		-- load test image
		local test_img = image.load(test_img_path)

		-- predict for test image
		local test_img_pred = predict(test_img)

		paths.mkdir(paths.concat('_save', 'test'))
		if t < 10 then
			for c = 1,5 do
				local img_save = test_img_pred[c]
				local file_name = string.format('test_%s_%s.jpg', test_num, labels[c])
				image.save(paths.concat('_save', 'test', file_name), img_save)
			end
		end

		local line = {test_num}
		for c = 1,5 do
			local feature = extract(test_img_pred[c])
			local pred = torch.dot(feature, coefs[c])
			pred = math.max(pred, 0)
			pred = string.format('%.0f', pred)

			table.insert(line, pred)
		end

		local rowwrite = table.concat(line, ',')
		fp:write(rowwrite .. '\n')
	end

	xlua.progress(#test_list, #test_list)
	fp:close()
end
