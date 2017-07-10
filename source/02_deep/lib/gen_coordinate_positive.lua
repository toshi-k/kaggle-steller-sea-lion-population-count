
------------------------------
-- function
------------------------------

function gen_coordinate_positive(coordinate, targets)
	local coordinate_positive

	coordinate_positive = {}
	coordinate_positive['adult_females'] = {}
	coordinate_positive['subadult_males'] = {}
	coordinate_positive['adult_males'] = {}
	coordinate_positive['pups'] = {}
	coordinate_positive['juveniles'] = {}

	for _, target in ipairs(targets) do
		for _, label in ipairs(labels) do
			for _, tb in ipairs(coordinate[tostring(target)][label]) do
				local lion_x = tb[1]
				local lion_y = tb[2]
				table.insert(coordinate_positive[label], {target, lion_x, lion_y})
			end
		end
	end

	return coordinate_positive
end
