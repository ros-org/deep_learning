all:
	mkdir -p build/lib
	mkdir -p build/bin
	$(MAKE) -C src	  all            # 执行src下面的Makefile中的all下面的代码;
	$(MAKE) -C TESTS	  all
# $(MAKE) -C BLL_TEST all
	
clean:
	$(MAKE) -C src	  clean
	$(MAKE) -C TESTS	  clean      # 执行src下面的Makefile中的clean下面的代码;
# $(MAKE) -C BLL_TEST clean
	
