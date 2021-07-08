import os
import utils
import torch
import torch.nn as nn
from tqdm import tqdm
from extract import extract
from test import do_eval


def train(args, epoch, model,
          train_loader, dev_loaders,
          summarizer, optimizer, scheduler):

    total_pred_loss, train_result = 0, None
    epoch_steps = int(args.total_steps / args.epochs)

    iterator = tqdm(enumerate(train_loader), desc='steps', total=epoch_steps)
    for step, batch in iterator:
        batch = map(lambda x: x.to(args.device), batch)
        token_ids, att_mask, single_pred_label, all_pred_label = batch
        pred_mask = utils.get_pred_mask(single_pred_label)

        model.train()
        model.zero_grad()

        pred_loss = model(input_ids=token_ids,
                          attention_mask=att_mask,
                          predicate_mask=pred_mask,
                          total_pred_labels=all_pred_label)
        total_pred_loss += pred_loss
        pred_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_result = total_pred_loss / (step + 1)
        if step > epoch_steps:
            break

        if step % 1000 == 0 and step != 0:
            dev_iter = zip(args.dev_data_path, args.dev_gold_path, dev_loaders)
            dev_results = list()
            total_sum = 0

            for dev_input, dev_gold, dev_loader in dev_iter:
                output_path = os.path.join(
                    args.save_path, f'epoch{epoch}_dev/step{step}')
                extract(args, model, dev_loader, output_path)
                dev_result = do_eval(output_path, dev_gold)

                utils.print_results(f"EPOCH{epoch} STEP{step} EVAL",
                                    dev_result, ["F1  ", "PREC", "REC ", "AUC "])
                total_sum += dev_result[0] + dev_result[-1]
                dev_result.append(dev_result[0] + dev_result[-1])
                dev_results += dev_result

            summarizer.save_results(
                [step] + train_result + dev_results + [total_sum])
            model_name = utils.set_model_name(total_sum, epoch, step)
            torch.save(model.state_dict(), os.path.join(
                args.save_path, model_name))

        if step % args.summary_step == 0 and step != 0:
            utils.print_results(f"EPOCH{epoch} STEP{step} TRAIN",
                                train_result, ["PRED LOSS", "ARG LOSS "])

    utils.print_results(f"EPOCH{epoch} TRAIN",
                        train_result, ["PRED LOSS", "ARG LOSS "])
    return train_result
