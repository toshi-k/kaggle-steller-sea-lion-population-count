
------------------------------
-- function
------------------------------

function load_train_list(path)
	local all_list = {}

	for str in paths.iterfiles(path) do
		local num = string.sub(str, 1, -5)
		table.insert(all_list, num)
	end

	print('\tnum train images (all): ' .. #all_list)
	print('\t\t' .. tostring(tablex.sub(all_list,1,10)) .. '...'
		.. tostring(tablex.sub(all_list,-10,Inf)))

	return all_list
end
