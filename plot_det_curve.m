function  plot_det_curve(tar_scores,nontar_scores, model)

plot_title = strcat('DET polt for ', model, ' model');
prior = 0.3;

plot_type = Det_Plot.make_plot_window_from_string('old');
plot_obj = Det_Plot(plot_type,plot_title);

plot_obj.set_system(tar_scores,nontar_scores,'sc');
plot_obj.plot_steppy_det({'b','LineWidth',2},' ');
plot_obj.plot_DR30_fa('c--','30 false alarms');
plot_obj.plot_DR30_miss('k--','30 misses');
plot_obj.plot_mindcf_point(prior,{'b*','MarkerSize',8},'mindcf');


plot_obj.display_legend();

fprintf('Look at the figure entitled ''DET plot example'' to see an example of a DET plot.\n');
