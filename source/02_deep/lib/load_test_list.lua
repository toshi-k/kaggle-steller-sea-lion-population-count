
------------------------------
-- function
------------------------------

function load_test_list(path)

	local test_list = {}

	local fp = io.open(path, 'r')

	-- remove header
	local str = fp:read()

	while str ~= nil do
		str = fp:read()
		if str ~= nil then
			local tb = utils.split(str, ',')
			table.insert(test_list, tb[1])
		end
	end
	fp:close()

	print('\tnum test: ' .. #test_list)
	print('\t\t' .. tostring(tablex.sub(test_list,1,10)) .. '...'
			.. tostring(tablex.sub(test_list,-10,Inf)))

	return test_list
end
