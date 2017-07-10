
------------------------------
-- function
------------------------------

function load_train_csv(path)
	local train_csv = {}

	local fp = io.open(path, 'r')

	-- remove header
	local str = fp:read()

	while str ~= nil do
		str = fp:read()
		if str ~= nil then
			local tb = utils.split(str, ',')
			-- test_id, adult_males, subadult_males, adult_females, juveniles, pups
			train_csv[tb[1]] = torch.Tensor{tb[2], tb[3], tb[4], tb[5], tb[6]}
		end
	end
	fp:close()

	return train_csv
end
