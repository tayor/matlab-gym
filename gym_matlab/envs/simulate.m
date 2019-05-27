function result = simulate(a, b)
    a = string(a);
    b = string(b);
    load_system('test');
    set_param('test/a', 'value', a);
    set_param('test/b', 'value', b);
    sim('test');
    result = y;
end
