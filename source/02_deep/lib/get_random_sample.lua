
------------------------------
-- library
------------------------------

require 'image'

------------------------------
-- function
------------------------------

local function int(value)
	return torch.round(value)
end

local function get_random_sample_from_target(target, patch_size)

	local train = image.load('../../dataset/Train/' .. target .. '.jpg')
	local train_dotted = image.load('../../dataset/TrainDotted/' .. target .. '.jpg')

	local scale = (math.random() * 0.6 - 0.3) + 1
	local scaled_patch_size = int(patch_size * scale)
	-- print(string.format('scale: %f scaled_patch_size: %d', scale, scaled_patch_size))

	local x = math.random(train:size(3) - scaled_patch_size)
	local y = math.random(train:size(2) - scaled_patch_size)
	-- print('target: ' .. target .. ' x: ' .. x .. ' y: ' .. y)

	local black = train_dotted[{{},{y, y+scaled_patch_size-1},{x, x+scaled_patch_size-1}}]:le(0.001):float():sum()

	if black < (scaled_patch_size * scaled_patch_size * 3 * 0.5) then
	-- if black < 10 then

		local output = torch.ByteTensor(5, 7, 7):fill(0)

		for label_id, label in ipairs(labels) do
			for _, tb in ipairs(coordinate[tostring(target)][label]) do

				local lion_x = tb[1]
				local lion_y = tb[2]

				if lion_x >= x and lion_x < x + scaled_patch_size and lion_y >= y and lion_y < y + scaled_patch_size then
					-- print("hit !")
					local lx = math.floor((lion_x - x) / 32.0 / scale) + 1
					local ly = math.floor((lion_y - y) / 32.0 / scale) + 1

					-- print('label_id: ' .. label_id .. ' lx: ' .. lx .. ' ly: ' .. ly)
					if ly >= 1 and ly <= output:size(2) and lx >= 1 and lx <= output:size(3) then
						output[{label_id, ly, lx}] = 1
					end
				end
			end
		end

		local input = train[{{},{y, y+scaled_patch_size-1},{x, x+scaled_patch_size-1}}]
		local input_dotted = train_dotted[{{},{y, y+scaled_patch_size-1},{x, x+scaled_patch_size-1}}]

		input = image.scale(input, patch_size, patch_size)
		input_dotted = image.scale(input_dotted, patch_size, patch_size)

		return input, output, input_dotted
	else
		return nil
	end
end

function get_random_sample(target_list, patch_size)

	local input, output, dotted

	while true do
		local target = target_list[math.random(#target_list)]
		input, output, dotted = get_random_sample_from_target(target, patch_size)
		if input ~= nil then break end
	end

	return input, output, dotted
end
